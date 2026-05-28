# Mouse ECG/SCG Binary Classification Pipeline v2

从原始长时程记录重建的 ECG/SCG 二分类流水线。
EF<50 阈值标签，基于质量的滑动窗口分段，1D ResNet CNN + SVM + XGBoost + LightGBM + 集成。

## 项目结构

```
├── ecg_segments/                   # ECG 片段 (945 CSV, 500Hz, ~4s)
├── scg_segments/                   # SCG 片段 (945 CSV, 500Hz, ~4s)
├── features_ecg.xlsx               # ECG 特征 (tsfresh + HRV/形态学, 50 维)
├── features_scg_fixed.xlsx         # SCG 特征 (tsfresh + S1/S2 心音, 787 维)
├── labels_for_ecg.xlsx             # ECG 标签 (EF<50 → 1, EF>=50 → 0)
├── labels_for_scg.xlsx             # SCG 标签
│
├── clean_raw_data.py               # Step 0: 标签筛选 + 文件名去中文
├── preprocess_raw.py               # Step 1: 质量分段 + 信号过滤
├── build_labels.py                 # Step 2: EF<50 标签重建
├── feature_extract.py              # Step 3: ECG 特征提取
├── scg_feature_extract_fixed.py    # Step 3: SCG 特征提取
├── model_train_final.py           # Step 4: 模型训练与评估
│
├── .gitignore
└── README.md
```

## 数据

从 315 个原始长时程记录（500Hz, ~1-2min/文件）中提取，来自 87 只小鼠的纵向追踪（2025.04-07, 8 时间点）。

| 属性 | 值 |
|------|-----|
| ECG 片段 | 945 (top-3 质量/鼠/日期) |
| SCG 片段 | 945 (top-3 质量/鼠/日期) |
| 段长度 | 2-6 秒 |
| 标签阈值 | EF < 50 → class 1 (心衰) |
| 类别分布 | 831:114 = 7.3:1 |

## 流水线

### Step 0: 数据清洗
- 根据 `labels.xlsx` 筛选有标签的文件
- 去除文件名中的中文字符和注释
- 统一为 `{MouseID}.csv` 格式

### Step 1: 信号分段
- 原始通道: ECG (ECG 列), SCG (AZ 加速度列)
- 带通滤波: ECG 0.5-100Hz, SCG 10-150Hz
- 滑动窗口: 4 秒窗口, 2 秒步长
- 质量评分: R 峰一致性、无削波、基线稳定、相邻窗口互相关
- Top-3 质量段/鼠/日期

### Step 2: 标签重建
- 标签查找: `labels.xlsx` 优先 → TAC Excel 文件回退
- EF < 50 → label=1, EF >= 50 → label=0

### Step 3: 特征提取
- ECG: tsfresh EfficientFCParameters + HRV (5 维) + 形态学 (4 维) → 50 维
- SCG: tsfresh + S1/S2 心音特征 (10 维) → 787 维

### Step 4: 模型训练
- 固定 20% holdout 测试集
- Repeated Stratified K-Fold (5-fold × 2)
- BorderlineSMOTE 在 CV 内训练集上应用
- 模型: LR ElasticNet, Linear SVM, RBF SVM, XGBoost, LightGBM, 1D ResNet CNN
- 加权集成

## 使用方法

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn tsfresh openpyxl
pip install torch optuna xgboost lightgbm

# 完整流水线（需要原始数据在 temp_data/ 中）
python clean_raw_data.py          # Step 0
python preprocess_raw.py          # Step 1
python build_labels.py            # Step 2
python feature_extract.py         # Step 3 ECG
python scg_feature_extract_fixed.py  # Step 3 SCG
python model_train_final.py       # Step 4
```

## 结果

### ECG (315 段, 63 测试, 55:8 类别比)

| 模型 | ACC | AUC | F1 | Precision | Recall | TP/FP |
|------|-----|-----|-----|-----------|--------|-------|
| **CNN 1D ResNet** | 0.889 | **0.648** | 0.364 | **0.667** | 0.250 | 2/1 |
| RBF SVM | 0.508 | 0.632 | 0.311 | 0.189 | **0.875** | 7/30 |
| **Ensemble** | 0.762 | 0.566 | 0.348 | 0.267 | 0.500 | 4/11 |
| LightGBM | 0.349 | 0.436 | 0.255 | 0.149 | **0.875** | 7/40 |

### SCG (315 段, 63 测试, 55:8 类别比)

| 模型 | ACC | AUC | F1 | Precision | Recall | TP/FP |
|------|-----|-----|-----|-----------|--------|-------|
| **CNN 1D ResNet** | 0.794 | **0.798** | **0.435** | **0.333** | 0.625 | 5/10 |
| LightGBM | 0.651 | 0.598 | 0.353 | 0.231 | 0.750 | 6/20 |
| RBF SVM | 0.619 | 0.585 | 0.333 | 0.214 | 0.750 | 6/22 |

### 对比旧数据（诚实评估）

| 指标 | 旧 (183段, EF<55) | 新 (315段, EF<50, 诚实) | 变化 |
|------|--------------------|--------------------------|------|
| ECG AUC | 0.671 | 0.648 | -3% |
| ECG F1 | 0.615 | 0.364 | 受极度不平衡影响 |
| SCG AUC | 0.634 (CV) | 0.798 | +26% |
| 数据量 | 183/303 | 315/315 | 统一规模 |

### 数据泄露警示

从同一小鼠同次录制的滑动窗口中提取多个段 → 随机分割将同源段分入 train/test → **AUC 从 0.648 虚高至 0.911**。修复方案：每 (mouse, date) 只保留 1 个质量最佳段，确保 0% 组重叠。
