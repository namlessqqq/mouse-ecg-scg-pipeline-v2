"""
Improved SCG feature extraction with mouse-appropriate S1/S2 detection parameters.
Mouse physiology: HR 400-650 bpm, cardiac cycle ~90-150ms, S1-S2 gap ~15-40ms.

Usage: python scg_feature_extract_fixed.py
Output: features_scg_fixed.xlsx
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import warnings
warnings.filterwarnings('ignore')


class MouseSCGExtractor:
    """SCG feature extractor tuned for MOUSE physiology (NOT human)."""

    def __init__(self, folder_path='scg_segments', sampling_rate=500, label_excel='labels_for_scg.xlsx'):
        self.folder_path = folder_path
        self.sampling_rate = sampling_rate
        self.label_excel = label_excel
        self.labels = []
        self.file_names = []
        self.signals = []
        self.X_features = None

        # Load labels from Excel
        label_df = pd.read_excel(label_excel)
        self.label_map = {}
        for _, row in label_df.iterrows():
            fname = str(row['filename']).strip()
            if fname.endswith('.csv'):
                fname = fname[:-4]
            self.label_map[fname] = str(row['label']).strip()

    def read_segments(self):
        print("=" * 60)
        print("步骤 1: 读取SCG片段数据 (小鼠生理参数)")
        print("=" * 60)

        files = sorted([f for f in os.listdir(self.folder_path)
                        if f.endswith('.csv') or f.endswith('.txt')])
        print(f"找到 {len(files)} 个SCG片段文件")

        data_list = []
        skipped = 0

        for file in files:
            file_path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(file_path, header=None)
                signal = df.values.flatten().astype(float)
                self.signals.append(signal)
                idx = len(self.labels)

                for t, value in enumerate(signal):
                    data_list.append({'id': idx, 'time': t, 'value': value})

                base_name = file.replace('.csv', '').replace('.txt', '')
                if base_name in self.label_map:
                    self.labels.append(self.label_map[base_name])
                    self.file_names.append(file)
                else:
                    print(f"  警告: {file} 在标签Excel中未找到，跳过")
                    self.signals.pop()
                    skipped += 1

            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
                continue

        df_tsfresh = pd.DataFrame(data_list)
        unique_labels = set(self.labels)
        print(f"成功读取 {len(self.labels)} 个样本, 跳过 {skipped}")
        print(f"类别分布: {pd.Series(self.labels).value_counts().to_dict()}")

        if len(unique_labels) < 2:
            raise ValueError(f"标签中只有一个类别，无法分类")

        return df_tsfresh

    def extract_features_tsfresh(self, df):
        print("\n" + "=" * 60)
        print("步骤 2: 使用TSfresh提取通用特征")
        print("=" * 60)

        self.X_features = extract_features(
            df,
            column_id='id',
            column_sort='time',
            column_value='value',
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=4,
            disable_progressbar=False
        )
        impute(self.X_features)
        print(f"提取了 {self.X_features.shape[1]} 个通用特征")
        return self.X_features

    def _envelope(self, signal):
        """Hilbert envelope with low-pass smoothing (mouse-appropriate cutoff)."""
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        # Lower cutoff for mouse: keep more envelope detail at shorter intervals
        b, a = butter(2, 0.15, btype='low')
        envelope_smooth = filtfilt(b, a, envelope)
        return envelope_smooth

    def _detect_s1_s2_mouse(self, signal):
        """Detect S1/S2 peaks in mouse SCG signals.
        Mouse physiology:
          - Heart rate: 400-650 bpm → RR interval: ~90-150 ms
          - S1-S2 interval (systole): ~15-40 ms
          - S2-next S1 interval (diastole): ~50-110 ms
        """
        nyquist = 0.5 * self.sampling_rate

        # Wider bandpass for mouse: 10-150 Hz (capture more of mouse heart sound spectrum)
        low = max(5.0 / nyquist, 0.01)
        high = min(150.0 / nyquist, 0.99)
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        envelope = self._envelope(filtered)

        # Mouse-appropriate peak detection:
        # Height: 0.15 * max envelope (slightly relaxed)
        height = 0.15 * np.max(envelope) if np.max(envelope) > 0 else 0.01
        # Distance: ~0.02s = 10 samples at 500Hz (mouse S1-S2 is very close)
        distance = max(int(0.015 * self.sampling_rate), 3)
        # Width: at least 5ms (mouse heart sounds are brief)
        width = max(0.005 * self.sampling_rate, 2)

        peaks, properties = find_peaks(envelope, height=height, distance=distance, width=width)

        if len(peaks) < 2:
            return [], [], [], []

        peak_times = peaks
        peak_amps = envelope[peaks]

        # Classify peaks as S1/S2 based on mouse-appropriate intervals
        # Mouse: S1-S2 gap ~15-40ms, S2-S1 gap ~50-110ms
        # At 500Hz: S1-S2 = 7-20 samples, S2-S1 = 25-55 samples

        s1_pos, s2_pos = [], []
        s1_amp, s2_amp = [], []

        for i, (t, amp) in enumerate(zip(peak_times, peak_amps)):
            if i == 0:
                s1_pos.append(t)
                s1_amp.append(amp)
                continue

            interval = (t - peak_times[i-1]) / self.sampling_rate  # seconds

            # Mouse S1-S2: 0.01-0.05s
            if 0.01 <= interval <= 0.06:
                s2_pos.append(t)
                s2_amp.append(amp)
            # Mouse S2-S1: >0.05s
            elif interval > 0.05:
                s1_pos.append(t)
                s1_amp.append(amp)
            # interval < 0.01s → noise, skip

        # Trim to matching pairs
        min_len = min(len(s1_pos), len(s2_pos))
        return s1_pos[:min_len], s2_pos[:min_len], s1_amp[:min_len], s2_amp[:min_len]

    def _calculate_heart_sound_features(self, signal):
        """Calculate mouse SCG heart sound features."""
        s1_pos, s2_pos, s1_amp, s2_amp = self._detect_s1_s2_mouse(signal)

        default_features = {
            'heart_rate_bpm': np.nan,
            's1_amplitude_mean': np.nan,
            's2_amplitude_mean': np.nan,
            's1_s2_amp_ratio': np.nan,
            's1_s2_interval_ms': np.nan,
            's2_s1_interval_ms': np.nan,
            's1_duration_ms': np.nan,
            's2_duration_ms': np.nan,
            'num_s1_detected': len(s1_pos),
            'num_s2_detected': len(s2_pos),
        }

        if len(s1_pos) < 1 or len(s2_pos) < 1:
            return default_features

        # Heart rate from S1-S1 intervals
        if len(s1_pos) >= 2:
            s1_intervals = np.diff(s1_pos) / self.sampling_rate
            mean_rr = np.mean(s1_intervals)
            heart_rate = 60.0 / mean_rr if mean_rr > 0 else np.nan
            # Sanity check: mouse HR should be 200-800 bpm
            if heart_rate < 100 or heart_rate > 900:
                heart_rate = np.nan
        else:
            # Estimate from single S1-S2-S1 pattern (if available)
            if len(s1_pos) >= 1 and len(s2_pos) >= 1:
                # Approximate: use S1-S2 interval * 3 as rough RR estimate
                s1s2 = abs(s2_pos[0] - s1_pos[0]) / self.sampling_rate
                rough_rr = s1s2 * 3.0
                heart_rate = 60.0 / rough_rr if rough_rr > 0 else np.nan
            else:
                heart_rate = np.nan

        s1_amp_mean = np.mean(s1_amp) if s1_amp else np.nan
        s2_amp_mean = np.mean(s2_amp) if s2_amp else np.nan
        amp_ratio = s1_amp_mean / s2_amp_mean if (s2_amp_mean and s2_amp_mean > 0) else np.nan

        # S1-S2 interval (systole)
        s1_s2_intervals = []
        for i in range(min(len(s1_pos), len(s2_pos))):
            s1_s2_intervals.append(abs(s2_pos[i] - s1_pos[i]) / self.sampling_rate * 1000.0)
        s1_s2_interval_ms = np.mean(s1_s2_intervals) if s1_s2_intervals else np.nan

        # S2-next S1 interval (diastole)
        s2_s1_intervals = []
        for i in range(min(len(s2_pos), len(s1_pos)-1)):
            s2_s1_intervals.append(abs(s1_pos[i+1] - s2_pos[i]) / self.sampling_rate * 1000.0)
        s2_s1_interval_ms = np.mean(s2_s1_intervals) if s2_s1_intervals else np.nan

        # Duration using envelope half-width (adapted for mouse)
        nyquist = 0.5 * self.sampling_rate
        low = max(5.0 / nyquist, 0.01)
        high = min(150.0 / nyquist, 0.99)
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        envelope = self._envelope(filtered)

        def get_durations(peak_positions):
            durations_ms = []
            for p in peak_positions:
                try:
                    peak_val = envelope[p]
                    if peak_val <= 0:
                        continue
                    half_height = peak_val / 2.0
                    left = max(0, p - int(0.03 * self.sampling_rate))
                    right = min(len(envelope)-1, p + int(0.03 * self.sampling_rate))
                    # Search within window
                    l, r = p, p
                    while l > left and envelope[l] > half_height:
                        l -= 1
                    while r < right and envelope[r] > half_height:
                        r += 1
                    width_samples = r - l
                    width_ms = width_samples / self.sampling_rate * 1000.0
                    durations_ms.append(width_ms)
                except Exception:
                    continue
            return np.mean(durations_ms) if durations_ms else np.nan

        s1_dur = get_durations(s1_pos)
        s2_dur = get_durations(s2_pos)

        return {
            'heart_rate_bpm': heart_rate,
            's1_amplitude_mean': s1_amp_mean,
            's2_amplitude_mean': s2_amp_mean,
            's1_s2_amp_ratio': amp_ratio,
            's1_s2_interval_ms': s1_s2_interval_ms,
            's2_s1_interval_ms': s2_s1_interval_ms,
            's1_duration_ms': s1_dur,
            's2_duration_ms': s2_dur,
            'num_s1_detected': len(s1_pos),
            'num_s2_detected': len(s2_pos),
        }

    def extract_heart_sound_features(self):
        print("\n" + "=" * 60)
        print("步骤 2b: 提取小鼠SCG生理特征 (改进S1/S2检测)")
        print("=" * 60)

        feature_list = []
        for idx, signal in enumerate(self.signals):
            features = self._calculate_heart_sound_features(signal)
            feature_list.append(features)

        hss_df = pd.DataFrame(feature_list)

        # Report NaN rates
        for col in hss_df.columns:
            nan_pct = hss_df[col].isna().mean()
            if nan_pct > 0.3:
                print(f"  WARNING: {col} NaN率={nan_pct:.1%}")

        print(f"提取了 {hss_df.shape[1]} 个SCG生理特征")

        self.X_features = pd.concat([self.X_features, hss_df], axis=1)
        print(f"总特征数变为 {self.X_features.shape[1]}")

    def save_to_excel(self, output_file='features_scg_fixed.xlsx'):
        print("\n" + "=" * 60)
        print(f"步骤 3: 保存特征到 {output_file}")
        print("=" * 60)

        result_df = pd.DataFrame({
            'filename': self.file_names,
            'label': self.labels
        })
        result_df = pd.concat([result_df, self.X_features.reset_index(drop=True)], axis=1)
        result_df.to_excel(output_file, index=False)
        print(f"成功保存特征到 {output_file}")

    def run(self, output_file='features_scg_fixed.xlsx'):
        df = self.read_segments()
        self.extract_features_tsfresh(df)
        self.extract_heart_sound_features()
        self.save_to_excel(output_file)


if __name__ == "__main__":
    extractor = MouseSCGExtractor('scg_segments', sampling_rate=500)
    extractor.run(output_file='features_scg_fixed.xlsx')
