import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import warnings

warnings.filterwarnings('ignore')

label_path = 'labels_for_ecg.xlsx'
input_path = 'ecg_segments'
output_path = 'features_ecg.xlsx'


class ECGFeatureExtractor:
    """ECG特征提取类（通用特征 + 心电专业特征 + 形态学特征 + 特征选择）"""

    def __init__(self, folder_path, sampling_rate=500, label_excel=label_path):
        self.folder_path = folder_path
        self.sampling_rate = sampling_rate
        self.label_excel = label_excel
        self.labels = []          # 每个片段的标签
        self.file_names = []       # 每个片段的文件名（内部使用，不保存到输出）
        self.ecg_signals = []      # 原始信号列表（numpy数组）
        self.X_features = None     # 最终特征矩阵

    def read_ecg_segments(self):
        """读取文件夹中的所有心电片段，保存信号并转换为tsfresh格式；
           同时从Excel文件最后一列读取标签（按文件顺序对应）"""
        print("=" * 60)
        print("步骤 1: 读取ECG片段数据")
        print("=" * 60)

        files = sorted([f for f in os.listdir(self.folder_path)
                        if f.endswith('.csv') or f.endswith('.txt')])
        print(f"找到 {len(files)} 个ECG片段文件")

        data_list = []   # 用于tsfresh的长格式数据

        # 1. 读取所有ECG信号及文件名
        for idx, file in enumerate(files):
            file_path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(file_path, header=None)
                ecg_signal = df.values.flatten().astype(float)
                self.ecg_signals.append(ecg_signal)
                self.file_names.append(file.split('.csv')[0])

                # 转换为tsfresh需要的格式：(id, time, value)
                for t, value in enumerate(ecg_signal):
                    data_list.append({
                        'id': idx,      # 使用当前文件的序号作为id
                        'time': t,
                        'value': value
                    })
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
                continue

        # 2. 从Excel文件读取标签（只取最后一列）
        print(f"从 {self.label_excel} 读取标签（最后一列）...")
        try:
            label_df = pd.read_excel(self.label_excel)
            # 取最后一列
            label_col = label_df.iloc[:, -1]
            self.labels = label_col.astype(str).tolist()

        except Exception as e:
            raise RuntimeError(f"读取标签文件失败: {e}")

        # ----- 标签检查1：非空且数量匹配 -----
        if len(self.labels) == 0:
            raise ValueError("标签列表为空，请检查Excel文件内容")
        if len(self.labels) != len(self.file_names):
            raise ValueError(
                f"标签数量 ({len(self.labels)}) 与ECG文件数量 ({len(self.file_names)}) 不一致！"
            )

        # ----- 标签检查2：至少包含两个类别（否则特征选择无法进行）-----
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"标签中只有一个类别 '{unique_labels}'，无法进行有监督特征选择。请提供至少两个类别的样本。"
            )
        print(f"标签类别: {unique_labels} (共{len(unique_labels)}类)")

        df_tsfresh = pd.DataFrame(data_list)

        print(f"成功读取 {len(self.labels)} 个样本")
        print(f"类别分布: {pd.Series(self.labels).value_counts().to_dict()}")

        return df_tsfresh

    def extract_features_tsfresh(self, df):
        """使用TSfresh提取通用特征"""
        print("\n" + "=" * 60)
        print("步骤 2: 使用TSfresh提取特征")
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

        # 填充缺失值
        impute(self.X_features)

        # ----- 特征矩阵检查1：是否为空 -----
        if self.X_features.empty:
            raise RuntimeError("提取的特征矩阵为空，请检查输入数据格式是否正确")
        # ----- 特征矩阵检查2：是否存在全NaN列（impute应该已经处理，但双重保障）-----
        nan_cols = self.X_features.columns[self.X_features.isna().all()].tolist()
        if nan_cols:
            print(f"警告：以下特征列全为NaN，将被删除: {nan_cols}")
            self.X_features = self.X_features.drop(columns=nan_cols)

        print(f"提取了 {self.X_features.shape[1]} 个特征，样本数 {self.X_features.shape[0]}")
        return self.X_features

    def _detect_r_peaks(self, signal):
        """
        对单个心电信号进行R波检测（带通滤波 + 峰值检测）
        返回R峰位置（样本点索引）和检测到的心跳数
        """
        # 带通滤波 0.5~40 Hz
        nyquist = 0.5 * self.sampling_rate
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        # 峰值检测
        height = 0.6 * np.std(filtered)
        distance = int(0.2 * self.sampling_rate)

        peaks, _ = find_peaks(filtered, height=height, distance=distance)
        return peaks, len(peaks)

    def _calculate_hrv_features(self, peaks):
        """
        根据R峰位置计算HRV特征
        返回字典：平均心率(bpm)、RR间期均值(ms)、标准差(ms)、RMSSD(ms)、心跳数
        """
        if len(peaks) < 2:
            return {
                'mean_hr': np.nan,
                'rr_mean_ms': np.nan,
                'rr_std_ms': np.nan,
                'rr_rmssd_ms': np.nan,
                'num_beats': len(peaks)
            }

        rr_intervals = np.diff(peaks) / self.sampling_rate
        rr_ms = rr_intervals * 1000.0

        mean_rr_sec = np.mean(rr_intervals)
        mean_hr = 60.0 / mean_rr_sec if mean_rr_sec > 0 else np.nan

        rr_mean_ms = np.mean(rr_ms)
        rr_std_ms = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        rr_rmssd_ms = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else np.nan

        return {
            'mean_hr': mean_hr,
            'rr_mean_ms': rr_mean_ms,
            'rr_std_ms': rr_std_ms,
            'rr_rmssd_ms': rr_rmssd_ms,
            'num_beats': len(peaks)
        }

    def _extract_morphology_features(self, signal, r_peaks):
        """
        提取心电形态学特征（基于第一个完整心跳）
        返回字典：QRS宽度(ms)、QT间期(ms)、T波幅度、ST段斜率等
        """
        # 若R峰少于2个，无法提取可靠形态学特征
        if len(r_peaks) < 2:
            return {
                'qrs_width_ms': np.nan,
                'qt_interval_ms': np.nan,
                't_amp': np.nan,
                'st_slope': np.nan
            }

        # 以第一个R峰为中心，向前向后搜索QRS起点和终点
        pre_window = int(0.08 * self.sampling_rate)   # 向前80ms
        post_window = int(0.12 * self.sampling_rate)  # 向后120ms
        r_idx = r_peaks[0]

        # 边界保护
        start = max(0, r_idx - pre_window)
        end = min(len(signal), r_idx + post_window)

        # 计算一阶差分，用于寻找QRS起点和终点
        diff_sig = np.diff(signal)
        diff_sig = np.convolve(diff_sig, np.ones(3)/3, mode='same')

        # 寻找QRS起点
        search_start = max(0, r_idx - pre_window)
        search_end = r_idx
        qrs_start = r_idx
        for i in range(search_end, search_start, -1):
            if i-1 >= 0 and diff_sig[i-1] <= 0 and diff_sig[i] > 0:
                qrs_start = i
                break

        # 寻找QRS终点
        search_start = r_idx
        search_end = min(len(signal)-1, r_idx + post_window)
        qrs_end = r_idx
        for i in range(search_start, search_end):
            if i+1 < len(diff_sig) and diff_sig[i] > 0 and diff_sig[i+1] <= 0:
                qrs_end = i+1
                break

        qrs_width_samples = qrs_end - qrs_start
        qrs_width_ms = qrs_width_samples / self.sampling_rate * 1000.0

        # 寻找T波峰值
        t_search_start = qrs_end + int(0.15 * self.sampling_rate)
        t_search_end = min(len(signal), qrs_end + int(0.5 * self.sampling_rate))
        if t_search_start < len(signal) and t_search_end > t_search_start:
            t_region = signal[t_search_start:t_search_end]
            if len(t_region) > 0:
                t_peak_idx = t_search_start + np.argmax(t_region)
                t_amp = signal[t_peak_idx]
            else:
                t_amp = np.nan
        else:
            t_amp = np.nan

        # QT间期
        if not np.isnan(t_amp) and t_peak_idx > qrs_start:
            qt_interval_ms = (t_peak_idx - qrs_start) / self.sampling_rate * 1000.0
        else:
            qt_interval_ms = np.nan

        # ST段斜率
        st_start = qrs_end + int(0.02 * self.sampling_rate)
        st_end = qrs_end + int(0.08 * self.sampling_rate)
        if st_end < len(signal) and st_start < len(signal):
            st_slope = (signal[st_end] - signal[st_start]) / ((st_end - st_start) / self.sampling_rate)
        else:
            st_slope = np.nan

        return {
            'qrs_width_ms': qrs_width_ms,
            'qt_interval_ms': qt_interval_ms,
            't_amp': t_amp,
            'st_slope': st_slope
        }

    def extract_ecg_features(self):
        """为每个心电片段提取专业ECG特征（HRV + 形态学），并合并到self.X_features中"""
        print("\n" + "=" * 60)
        print("步骤 2b: 提取心电专业特征 (HRV + 形态学)")
        print("=" * 60)

        hrv_list = []
        morph_list = []

        for idx, signal in enumerate(self.ecg_signals):
            peaks, _ = self._detect_r_peaks(signal)
            hrv_feats = self._calculate_hrv_features(peaks)
            morph_feats = self._extract_morphology_features(signal, peaks)
            hrv_list.append(hrv_feats)
            morph_list.append(morph_feats)

        hrv_df = pd.DataFrame(hrv_list)
        morph_df = pd.DataFrame(morph_list)
        ecg_df = pd.concat([hrv_df, morph_df], axis=1)

        print(f"提取了 {ecg_df.shape[1]} 个心电专业特征 (HRV + 形态学)")

        # 合并到总特征矩阵
        self.X_features = pd.concat([self.X_features, ecg_df], axis=1)
        print(f"总特征数变为 {self.X_features.shape[1]}")

    def select_features_tsfresh(self, target_labels, variance_threshold=0.01, corr_threshold=0.95, k=41):
        """
        改进的特征选择：手工ECG特征强制保留，通用特征进行初步过滤
        """
        print("\n" + "=" * 60)
        print("步骤 2c: 自定义特征选择 (保留手工ECG特征)")
        print("=" * 60)

        # ========== 输入严格检查 ==========
        if self.X_features is None:
            raise RuntimeError("特征矩阵 self.X_features 尚未提取，请先运行 extract_features_tsfresh() 和 extract_ecg_features()")
        if self.X_features.empty:
            raise RuntimeError("特征矩阵为空，无法进行特征选择")

        if target_labels is None or len(target_labels) == 0:
            raise ValueError("标签列表为空，无法进行有监督特征选择")
        if len(target_labels) != self.X_features.shape[0]:
            raise ValueError(
                f"标签数量 ({len(target_labels)}) 与特征矩阵行数 ({self.X_features.shape[0]}) 不一致！"
            )
        unique_labels = set(target_labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"标签中只有一个类别 '{unique_labels}'，特征选择需要至少两个类别。请检查标签数据。"
            )

        non_nan_cols = self.X_features.columns[~self.X_features.isna().all()].tolist()
        if len(non_nan_cols) == 0:
            raise RuntimeError("特征矩阵所有列均为NaN，无法进行特征选择")
        self.X_features = self.X_features[non_nan_cols]

        # 1. 分离手工ECG特征（通过列名关键词）
        ecg_keywords = ['mean_hr', 'rr_', 'num_beats', 'qrs_width', 'qt_interval', 't_amp', 'st_slope']
        ecg_cols = [col for col in self.X_features.columns if any(kw in col for kw in ecg_keywords)]
        other_cols = [col for col in self.X_features.columns if col not in ecg_cols]

        print(f"手工ECG特征数: {len(ecg_cols)}")
        print(f"其他通用特征数: {len(other_cols)}")

        if len(other_cols) == 0:
            print("无通用特征需要筛选，保留全部手工特征")
            return

        X_other = self.X_features[other_cols].copy()

        # 3.1 移除方差过低的特征
        var_vals = X_other.var()
        low_var_cols = var_vals[var_vals < variance_threshold].index.tolist()
        X_other.drop(columns=low_var_cols, inplace=True)
        print(f"移除低方差特征 {len(low_var_cols)} 个，剩余 {X_other.shape[1]} 个")

        if X_other.shape[1] == 0:
            print("通用特征全部被低方差过滤，仅保留手工特征")
            self.X_features = self.X_features[ecg_cols]
            return

        # 3.2 移除高相关性特征对
        corr_matrix = X_other.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
        X_other.drop(columns=to_drop, inplace=True)
        print(f"移除高相关特征 {len(to_drop)} 个，剩余 {X_other.shape[1]} 个")

        if X_other.shape[1] == 0:
            print("通用特征全部被高相关过滤，仅保留手工特征")
            self.X_features = self.X_features[ecg_cols]
            return

        # 3.3 基于互信息选择 top-k 特征
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(target_labels)

        X_other = X_other.fillna(X_other.mean())

        if k > 0 and k < X_other.shape[1]:
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X_other, y)
            selected_mask = selector.get_support()
            selected_cols = X_other.columns[selected_mask].tolist()
            X_other = X_other[selected_cols]
            print(f"基于互信息选择 {len(selected_cols)} 个特征")
        else:
            print("保留所有剩余通用特征")

        # 4. 合并手工特征 + 筛选后的通用特征
        self.X_features = pd.concat([self.X_features[ecg_cols], X_other], axis=1)
        print(f"最终特征总数: {self.X_features.shape[1]} (手工 {len(ecg_cols)} + 通用 {X_other.shape[1]})")

    def save_to_excel(self, output_file='extracted_features.xlsx'):
        """将提取的特征保存到Excel文件（第一列：文件名，第二列：标签，后续为特征）"""
        print("\n" + "=" * 60)
        print(f"步骤 3: 保存特征到 {output_file}")
        print("=" * 60)

        # 构建DataFrame：第一列文件名，第二列标签，然后特征
        result_df = pd.DataFrame({
            'filename': self.file_names,
            'label': self.labels
        })
        result_df = pd.concat([result_df, self.X_features.reset_index(drop=True)], axis=1)

        try:
            result_df.to_excel(output_file, index=False)
            print(f"成功保存特征到 {output_file} (第一列:文件名, 第二列:标签)")
        except Exception as e:
            print(f"保存失败: {e}")
            print("提示: 保存为xlsx格式需要安装 'openpyxl' 库，请运行: pip install openpyxl")

    def run(self, output_file='extracted_features.xlsx', do_feature_selection=True, p_threshold=0.5):
        """运行完整流程：读取 -> 提取tsfresh特征 -> 提取心电专业特征 -> 特征选择 -> 保存"""
        df = self.read_ecg_segments()
        self.extract_features_tsfresh(df)
        self.extract_ecg_features()

        if do_feature_selection:
            try:
                self.select_features_tsfresh(self.labels)
            except (ValueError, RuntimeError) as e:
                print(f"\n!!! 特征选择跳过: {e}")
                print("将保留所有特征（不进行特征选择）")
        else:
            print("\n跳过了特征选择步骤")

        self.save_to_excel(output_file)


if __name__ == "__main__":
    extractor = ECGFeatureExtractor(input_path, sampling_rate=500)
    extractor.run(output_file=output_path, do_feature_selection=True, p_threshold=0.5)