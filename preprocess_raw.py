"""
Step 1: Raw signal segmentation with quality-based filtering
=============================================================
Reads cleaned raw CSV files, extracts ECG/SCG channels, applies quality-aware
sliding window segmentation, and outputs valid segments.

Signal quality assessment uses:
  - Cross-correlation between consecutive windows (similarity check)
  - R-peak consistency (ECG) / envelope stability (SCG)
  - Absence of saturation, clipping, and abrupt baseline jumps
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert, resample
from collections import defaultdict

# =========================  CONFIG  =========================
RAW_DIR = 'temp_data/raw_clean'
ECG_OUT = 'ecg_segments'
SCG_OUT = 'scg_segments'
SR = 500                      # Sampling rate (Hz)
WINDOW_SEC = 4.0              # Segment window (seconds)
STRIDE_SEC = 2.0              # Sliding stride (seconds)
ECG_BANDPASS = (0.5, 100.0)   # ECG bandpass (Hz)
SCG_BANDPASS = (10.0, 150.0)  # SCG bandpass (Hz)
QUALITY_THRESH = 0.3          # Minimum quality score (0-1)
MIN_SEGMENT_SAMPLES = int(SR * 0.5)  # Minimum segment = 0.5 sec
MAX_SEGMENT_SEC = 6.0         # Maximum segment = 6 sec after merging
DATE_DIRS = ['4.18', '4.29', '4.30', '5.14', '5.28', '6.11', '6.25', '7.10']


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.01)
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(signal, lowcut, highcut, fs):
    """Apply zero-phase bandpass filter. Returns filtered signal same length."""
    if len(signal) < 10:
        return signal
    b, a = butter_bandpass(lowcut, highcut, fs)
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def resample_to_fixed(signal, target_sr, orig_sr):
    """Resample signal to exact target sampling rate."""
    if orig_sr == target_sr:
        return signal
    target_len = int(len(signal) * target_sr / orig_sr)
    return resample(signal, target_len)


def detect_r_peaks(signal, fs):
    """Detect R-peaks in ECG signal. Returns peak indices."""
    if len(signal) < int(fs * 0.2):  # Need at least 0.2 sec
        return np.array([])
    height = 0.5 * np.std(signal)
    distance = int(0.06 * fs)  # Min 60ms between peaks (mouse HR up to 1000 bpm)
    try:
        peaks, _ = find_peaks(signal, height=height, distance=distance)
        return peaks
    except Exception:
        return np.array([])


def ecg_quality_score(signal, fs, prev_signal=None):
    """Score an ECG window (0-1). Higher = better quality.

    Criteria:
      - Has detectable R-peaks (heartbeats)
      - R-R intervals are stable (not arrhythmic/noisy)
      - Signal not saturated (clipped at rails)
      - No abrupt baseline jumps
      - Similar to previous window (cross-correlation)
    """
    score = 0.0
    n = len(signal)

    # 1. R-peak detection and consistency (weight: 0.4)
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) >= 3:
        score += 0.2
        rr = np.diff(peaks) / fs * 1000  # RR intervals in ms
        if len(rr) >= 2:
            cv_rr = np.std(rr) / (np.mean(rr) + 1e-10)  # Coeff of variation
            if cv_rr < 0.5:   # Reasonably consistent rhythm
                score += 0.1
            if cv_rr < 0.3:   # Very consistent
                score += 0.1
    elif len(peaks) >= 1:
        score += 0.1

    # 2. Amplitude range check (weight: 0.2)
    sig_range = np.max(signal) - np.min(signal)
    sig_std = np.std(signal)
    if sig_std > 1e-10 and sig_range > 0:
        # Check for saturation: too many samples at same extreme value
        upper_sat = np.sum(np.abs(signal - np.max(signal)) < 1e-8)
        lower_sat = np.sum(np.abs(signal - np.min(signal)) < 1e-8)
        sat_ratio = (upper_sat + lower_sat) / n
        if sat_ratio < 0.05:
            score += 0.1
        if sat_ratio < 0.01:
            score += 0.1

    # 3. Baseline stability (weight: 0.2)
    if n > fs:  # At least 1 second
        # Split into sub-windows, check mean drift
        n_sub = min(4, n // fs)
        sub_len = n // n_sub
        means = [np.mean(signal[i*sub_len:(i+1)*sub_len]) for i in range(n_sub)]
        mean_drift = np.std(means) / (sig_std + 1e-10)
        if mean_drift < 0.5:
            score += 0.1
        if mean_drift < 0.2:
            score += 0.1

    # 4. Similarity to previous window (weight: 0.2)
    if prev_signal is not None and len(prev_signal) == n:
        corr = np.corrcoef(signal, prev_signal)[0, 1]
        if not np.isnan(corr) and corr > 0.3:
            score += 0.1
        if not np.isnan(corr) and corr > 0.6:
            score += 0.1

    return score


def scg_quality_score(signal, fs, prev_signal=None):
    """Score an SCG window (0-1). Higher = better quality.

    Criteria:
      - Signal has periodic structure (envelope peaks)
      - Envelope stability across window
      - Not saturated
      - No abrupt jumps
      - Similar to previous window
    """
    score = 0.0
    n = len(signal)

    # 1. Envelope-based periodicity (weight: 0.4)
    if n > 20:
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        # Low-pass the envelope for peak detection
        env_peaks, _ = find_peaks(envelope, height=0.3 * np.max(envelope),
                                   distance=int(0.06 * fs))
        if len(env_peaks) >= 2:
            score += 0.2
            intervals = np.diff(env_peaks) / fs * 1000
            if len(intervals) >= 1:
                cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                if cv < 0.6:
                    score += 0.1
                if cv < 0.3:
                    score += 0.1

    # 2. Amplitude range (weight: 0.2)
    sig_range = np.max(signal) - np.min(signal)
    sig_std = np.std(signal)
    if sig_std > 1e-10 and sig_range > 0:
        upper_sat = np.sum(np.abs(signal - np.max(signal)) < 1e-8)
        lower_sat = np.sum(np.abs(signal - np.min(signal)) < 1e-8)
        sat_ratio = (upper_sat + lower_sat) / n
        if sat_ratio < 0.05:
            score += 0.1
        if sat_ratio < 0.02:
            score += 0.1

    # 3. Baseline stability (weight: 0.2)
    if n > fs:
        n_sub = min(4, n // fs)
        sub_len = n // n_sub
        means = [np.mean(signal[i*sub_len:(i+1)*sub_len]) for i in range(n_sub)]
        mean_drift = np.std(means) / (sig_std + 1e-10)
        if mean_drift < 0.5:
            score += 0.1
        if mean_drift < 0.2:
            score += 0.1

    # 4. Cross-correlation with previous window (weight: 0.2)
    if prev_signal is not None and len(prev_signal) == n:
        corr = np.corrcoef(signal, prev_signal)[0, 1]
        if not np.isnan(corr) and corr > 0.2:
            score += 0.1
        if not np.isnan(corr) and corr > 0.5:
            score += 0.1

    return score


def segment_signal(signal, fs, window_sec, stride_sec, quality_fn, max_seg_sec=MAX_SEGMENT_SEC):
    """Sliding window segmentation with quality filtering and overlap merging.

    Returns list of (start_sample, end_sample, quality_score) for valid segments.
    """
    win_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    max_seg_samples = int(max_seg_sec * fs)

    if len(signal) < win_samples:
        return []

    windows = []
    prev_win = None
    for start in range(0, len(signal) - win_samples + 1, stride_samples):
        end = start + win_samples
        win = signal[start:end]
        score = quality_fn(win, fs, prev_win)
        if score >= QUALITY_THRESH:
            windows.append((start, end, score))
        prev_win = win

    if not windows:
        return []

    # Merge overlapping/adjacent high-quality windows
    merged = []
    curr_start, curr_end, curr_score = windows[0]
    for start, end, score in windows[1:]:
        if start <= curr_end and (end - curr_start) <= max_seg_samples:
            # Overlapping: extend current segment
            curr_end = max(curr_end, end)
            curr_score = max(curr_score, score)
        else:
            if curr_end - curr_start >= MIN_SEGMENT_SAMPLES:
                merged.append((curr_start, curr_end, curr_score))
            curr_start, curr_end, curr_score = start, end, score

    # Don't forget the last segment
    if curr_end - curr_start >= MIN_SEGMENT_SAMPLES:
        merged.append((curr_start, curr_end, curr_score))

    return merged


def process_all():
    """Main processing loop over all cleaned raw files."""
    os.makedirs(ECG_OUT, exist_ok=True)
    os.makedirs(SCG_OUT, exist_ok=True)

    # Clean old segments
    for f in os.listdir(ECG_OUT):
        os.remove(os.path.join(ECG_OUT, f))
    for f in os.listdir(SCG_OUT):
        os.remove(os.path.join(SCG_OUT, f))

    metadata = {'ecg_segments': [], 'scg_segments': []}
    total_ecg, total_scg = 0, 0

    for date_dir in DATE_DIRS:
        src_dir = os.path.join(RAW_DIR, date_dir)
        if not os.path.isdir(src_dir):
            continue

        csv_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.csv')])
        print(f"\n{'='*50}")
        print(f"Processing {date_dir}/ — {len(csv_files)} files")
        print(f"{'='*50}")

        for fname in csv_files:
            src_path = os.path.join(src_dir, fname)
            try:
                df = pd.read_csv(src_path)
            except Exception as e:
                print(f"  SKIP {fname}: read error - {e}")
                continue

            # Extract raw channels
            if 'ECG' not in df.columns or 'AZ' not in df.columns:
                print(f"  SKIP {fname}: missing ECG/AZ columns")
                continue

            ecg_raw = df['ECG'].values.astype(float)
            scg_raw = df['AZ'].values.astype(float)

            # Remove NaN
            ecg_raw = np.nan_to_num(ecg_raw, nan=np.nanmedian(ecg_raw) if len(ecg_raw) > 0 else 0)
            scg_raw = np.nan_to_num(scg_raw, nan=np.nanmedian(scg_raw) if len(scg_raw) > 0 else 0)

            # Remove DC offset
            if len(ecg_raw) > 0:
                ecg_raw = ecg_raw - np.mean(ecg_raw)
            if len(scg_raw) > 0:
                scg_raw = scg_raw - np.mean(scg_raw)

            # Bandpass filter the raw signals
            ecg_filt = bandpass_filter(ecg_raw, ECG_BANDPASS[0], ECG_BANDPASS[1], SR)
            scg_filt = bandpass_filter(scg_raw, SCG_BANDPASS[0], SCG_BANDPASS[1], SR)

            mouse_id = fname.replace('.csv', '')
            date_prefix = date_dir.replace('.', '_')

            # ---- ECG segmentation ----
            ecg_segs = segment_signal(ecg_filt, SR, WINDOW_SEC, STRIDE_SEC, ecg_quality_score)
            for idx, (start, end, score) in enumerate(ecg_segs):
                seg_signal = ecg_filt[start:end]
                seg_name = f"{date_prefix}_{mouse_id}_seg{idx}.csv"
                seg_path = os.path.join(ECG_OUT, seg_name)
                np.savetxt(seg_path, seg_signal, delimiter=',')
                metadata['ecg_segments'].append({
                    'file': seg_name, 'mouse': mouse_id, 'date': date_dir,
                    'start': start, 'end': end, 'quality': round(score, 3),
                    'length_sec': (end - start) / SR
                })
                total_ecg += 1

            # ---- SCG segmentation ----
            scg_segs = segment_signal(scg_filt, SR, WINDOW_SEC, STRIDE_SEC, scg_quality_score)
            # Use shorter windows for SCG (1 sec) since SCG cycles are much shorter
            scg_segs_short = segment_signal(scg_filt, SR, 1.0, 0.5, scg_quality_score, max_seg_sec=2.0)
            all_scg = scg_segs + scg_segs_short
            # Deduplicate SCG segments (prefer longer ones)
            all_scg.sort(key=lambda x: (x[0], -(x[1]-x[0])))  # Sort by start, then longest first
            deduped_scg = []
            for seg in all_scg:
                # Check overlap with already selected segments
                overlap = False
                for existing in deduped_scg:
                    if seg[0] < existing[1] and seg[1] > existing[0]:
                        overlap = True
                        break
                if not overlap:
                    deduped_scg.append(seg)

            for idx, (start, end, score) in enumerate(deduped_scg):
                seg_signal = scg_filt[start:end]
                seg_name = f"{date_prefix}_{mouse_id}_seg{idx}.csv"
                seg_path = os.path.join(SCG_OUT, seg_name)
                np.savetxt(seg_path, seg_signal, delimiter=',')
                metadata['scg_segments'].append({
                    'file': seg_name, 'mouse': mouse_id, 'date': date_dir,
                    'start': start, 'end': end, 'quality': round(score, 3),
                    'length_sec': (end - start) / SR
                })
                total_scg += 1

            if ecg_segs or deduped_scg:
                print(f"  {fname}: {len(ecg_segs)} ECG + {len(deduped_scg)} SCG segments")

    # Save metadata
    with open('segment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"DONE: {total_ecg} ECG segments → {ECG_OUT}/")
    print(f"      {total_scg} SCG segments → {SCG_OUT}/")
    print(f"      Metadata → segment_metadata.json")


if __name__ == '__main__':
    process_all()
