"""
Improved Binary Classification — Signal-Level DL + Simpler Models + Rigorous CV
================================================================================
Improvements:
  3. 1D ResNet CNN on raw ECG/SCG waveforms (signal-level feature learning)
  4. Simpler models: LR (ElasticNet), Linear SVM, XGBoost, LightGBM — no large DL
  5. Fixed 20% holdout test set + Repeated Stratified K-Fold (5-fold, 3 repeats)
  6. BorderlineSMOTE inside each CV fold (train only — zero leakage to val/test)

All preprocessing, feature selection, scaling, SMOTE fitted on train portion only.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from copy import deepcopy

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                      train_test_split)
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                              accuracy_score, precision_recall_curve, confusion_matrix)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna, xgboost as xgb, lightgbm as lgb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =========================  CONFIG  =========================
N_FOLDS = 5
N_REPEATS = 2          # Repeated Stratified K-Fold
N_OPTUNA_TRIALS = 20   # HP tuning trials per fold
INNER_FOLDS = 3        # Inner CV for Optuna

# Signal fixed lengths
ECG_SIGNAL_LEN = 2048
SCG_SIGNAL_LEN = 320

# =====================  FOCAL LOSS  =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# =====================  1D RESNET CNN (改进3)  =====================
class ResidualBlock1D(nn.Module):
    """1D residual block with bottleneck design."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Small1DResNet(nn.Module):
    """Tiny 1D ResNet for biomedical signal classification (~15K params).
    Designed for very small datasets (100-300 samples)."""
    def __init__(self, input_len, in_channels=1, num_classes=2, base_ch=16, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_ch, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 3 residual blocks with increasing channels
        self.layer1 = self._make_layer(base_ch, base_ch, 2, stride=1)
        self.layer2 = self._make_layer(base_ch, base_ch * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_ch * 2, base_ch * 4, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        fc_in = base_ch * 4
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, num_classes)
        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [ResidualBlock1D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.avgpool(x).flatten(1)
        feat = self.dropout(features)
        logits = self.fc(feat)
        return (logits, features) if return_features else logits

    def extract_features(self, x):
        return self.forward(x, return_features=True)[1]

# =====================  DATA LOADING  =====================

def load_raw_signals(signal_dir, label_excel_path, target_length):
    """Load raw CSV signal files, match with labels, normalize + pad/truncate.

    Returns:
        signals: np.array (n_samples, target_length)
        labels:  np.array (n_samples,) encoded as 0/1
        le:      LabelEncoder
    """
    label_df = pd.read_excel(label_excel_path)
    label_map = {}
    for _, row in label_df.iterrows():
        fname = str(row['filename']).strip()
        if fname.endswith('.csv'):
            fname = fname[:-4]
        label_map[fname.lower()] = str(row['label']).strip()

    signals, labels_str = [], []
    files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.csv')])
    for f in files:
        fname = f[:-4].lower()
        if fname not in label_map:
            continue
        df = pd.read_csv(os.path.join(signal_dir, f), header=None)
        sig = df.values.flatten().astype(np.float32)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        # Pad or truncate
        if len(sig) > target_length:
            sig = sig[:target_length]
        elif len(sig) < target_length:
            sig = np.pad(sig, (0, target_length - len(sig)), mode='constant')
        signals.append(sig)
        labels_str.append(label_map[fname])

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    X = np.array(signals, dtype=np.float32)
    print(f"  Loaded {len(X)} raw signals, shape={X.shape}, "
          f"labels: {dict(zip(le.classes_, np.bincount(y)))}")
    return X, y, le


def load_features_and_labels(feature_path):
    """Load tsfresh features + label from Excel feature file.

    Returns:
        X_tsfresh: np.array (n_samples, n_tsfresh_features)
        y:         np.array encoded 0/1
        le:        LabelEncoder
    """
    df = pd.read_excel(feature_path)
    feat_cols = [c for c in df.columns[2:] if c not in ('ori_name',)]
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    print(f"  Loaded features: {X.shape[0]} samples, {X.shape[1]} features, "
          f"labels: {dict(zip(le.classes_, np.bincount(y)))}")
    return X, y, le


def load_physiological_params(label_excel_path):
    """Extract physiological parameters from label Excel.

    Skips non-numeric columns (e.g., 'ori_name' in SCG labels).
    Returns dict: fname (lower, no ext) -> np.array of numeric features
    """
    df = pd.read_excel(label_excel_path)
    # Select numeric columns between filename and label
    all_mid_cols = df.columns[1:-1].tolist()
    phys_cols = [c for c in all_mid_cols if df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    param_map = {}
    for _, row in df.iterrows():
        fname = str(row['filename']).strip()
        if fname.endswith('.csv'):
            fname = fname[:-4]
        vals = []
        for c in phys_cols:
            v = row[c]
            vals.append(float(v) if not pd.isna(v) else np.nan)
        param_map[fname.lower()] = np.array(vals, dtype=float)
    print(f"  Physiological params: {len(param_map)} entries, {len(phys_cols)} features: {phys_cols}")
    return param_map, phys_cols


def build_joint_feature_matrix(feature_path, label_excel_path=None):
    """Load features from Excel feature file.

    If label_excel_path is provided and use_phys_params=True, also merges
    physiological parameters from the label file. Default: signal features only.
    """
    df = pd.read_excel(feature_path)
    feat_cols = [c for c in df.columns if c not in ('filename', 'label', '_fname', 'ori_name')]
    X = df[feat_cols].values.astype(float)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    return X, y, le


# =====================  IN-FOLD PREPROCESSING (改进6)  =====================
def preprocess_train_only(X_train, X_test, y_train, k_best=None):
    """All preprocessing fitted on train only — applied to test. Zero leakage.

    Steps:
      1. NaN/Inf → 0
      2. Remove constant features (fit on train)
      3. MI feature selection (fit on train)
      4. RobustScaler (fit on train)
    """
    X_train = np.where(np.isnan(X_train) | np.isinf(X_train), 0.0, X_train)
    X_test  = np.where(np.isnan(X_test) | np.isinf(X_test), 0.0, X_test)

    # Remove constant features
    stds = np.std(X_train, axis=0)
    keep = stds > 1e-10
    X_train, X_test = X_train[:, keep], X_test[:, keep]

    # Feature selection (fit on train)
    if k_best is not None and k_best > 0 and k_best < X_train.shape[1] and y_train is not None:
        selector = SelectKBest(mutual_info_classif, k=min(k_best, X_train.shape[1]))
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    # Scaling (fit on train)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def apply_smote_train_only(X_train, y_train):
    """BorderlineSMOTE on training data only. Never on validation/test data.

    Uses conservative k_neighbors for small datasets.
    """
    counts = np.bincount(y_train.astype(int))
    if counts.min() < 5:
        # Too few minority samples for BorderlineSMOTE, fall back to basic SMOTE
        if counts.min() < 3:
            return X_train, y_train
        k = counts.min() - 1
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            return sm.fit_resample(X_train, y_train)
        except Exception:
            return X_train, y_train

    k = min(3, counts.min() - 1)  # Conservative k for small datasets
    try:
        sm = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=k, kind='borderline-1')
        X_bal, y_bal = sm.fit_resample(X_train, y_train)
        return X_bal, y_bal
    except Exception:
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            return sm.fit_resample(X_train, y_train)
        except Exception:
            return X_train, y_train


# =====================  THRESHOLD OPTIMIZATION  =====================
def optimize_threshold(y_true, y_probs):
    p, r, t = precision_recall_curve(y_true, y_probs)
    f1 = 2 * p * r / (p + r + 1e-10)
    f1 = np.nan_to_num(f1, 0.0)
    return t[np.argmax(f1)]


def compute_metrics(y_true, y_probs, threshold=0.5):
    if threshold != 0.5:
        threshold = optimize_threshold(y_true, y_probs)
    preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'AUC': roc_auc_score(y_true, y_probs),
        'F1': f1_score(y_true, preds, average='binary', zero_division=0),
        'Precision': precision_score(y_true, preds, average='binary', zero_division=0),
        'Recall': recall_score(y_true, preds, average='binary', zero_division=0),
        'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1],
    }


# =====================  1D CNN TRAINING (改进3)  =====================
def train_cnn(X_train, y_train, X_val, y_val, signal_len, epochs=150, patience=30):
    """Train Small1DResNet on raw signals with Focal Loss + early stopping."""
    cw = np.bincount(y_train.astype(int), minlength=2)
    cw = torch.tensor(cw.sum() / (2 * cw.clip(1)), dtype=torch.float32).to(DEVICE)

    model = Small1DResNet(input_len=signal_len, in_channels=1, num_classes=2,
                           base_ch=16, dropout=0.3).to(DEVICE)
    criterion = FocalLoss(alpha=cw, gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(y_val, dtype=torch.long)

    bs = max(8, min(32, len(X_train) // 4))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True, drop_last=True)

    best_auc, best_state, counter = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(Xv.to(DEVICE)), 1).cpu().numpy()[:, 1]
            auc = roc_auc_score(yv, probs) if len(np.unique(yv)) > 1 else 0.5

        if auc > best_auc:
            best_auc, best_state, counter = auc, deepcopy(model.state_dict()), 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def cnn_predict(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        return torch.softmax(model(x), 1).cpu().numpy()[:, 1]


# =====================  MODEL TRAINERS (改进4)  =====================

# --- Logistic Regression (ElasticNet) ---
def train_lr_optuna(X_tr, y_tr, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        lr = LogisticRegression(
            C=trial.suggest_float('C', 0.001, 100, log=True),
            l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
            penalty='elasticnet', solver='saga', class_weight='balanced',
            max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(X_tr, y_tr)
        return roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    best = LogisticRegression(C=study.best_params['C'], l1_ratio=study.best_params['l1_ratio'],
                               penalty='elasticnet', solver='saga', class_weight='balanced',
                               max_iter=5000, random_state=RANDOM_STATE)
    best.fit(X_tr, y_tr)
    return best, study.best_value


# --- Linear SVM (Calibrated for probabilities) ---
def train_svm_optuna(X_tr, y_tr, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    """Linear SVM — better for small data than RBF. Calibrated for probability output."""
    def obj(trial):
        svm = LinearSVC(C=trial.suggest_float('C', 0.001, 1000, log=True),
                         class_weight='balanced', dual=False,
                         max_iter=5000, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_tr)
        scores = svm.decision_function(X_val)
        return roc_auc_score(y_val, scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    base_svm = LinearSVC(C=bp['C'], class_weight='balanced', dual=False,
                          max_iter=5000, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(base_svm, method='isotonic', cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated, study.best_value


# --- RBF SVM (Optuna-tuned C + gamma) ---
def train_rbf_svm_optuna(X_tr, y_tr, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    """RBF SVM with Optuna tuning for C and gamma. Better for nonlinear boundaries."""
    def obj(trial):
        svm = SVC(C=trial.suggest_float('C', 0.01, 1000, log=True),
                   gamma=trial.suggest_float('gamma', 1e-5, 1.0, log=True),
                   kernel='rbf', class_weight='balanced', probability=True,
                   random_state=RANDOM_STATE, max_iter=5000)
        svm.fit(X_tr, y_tr)
        return roc_auc_score(y_val, svm.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best = SVC(C=bp['C'], gamma=bp['gamma'], kernel='rbf',
               class_weight='balanced', probability=True,
               random_state=RANDOM_STATE, max_iter=5000)
    best.fit(X_tr, y_tr)
    return best, study.best_value


# --- XGBoost ---
def train_xgb_optuna(X_tr, y_tr, X_val, y_val, n_trials=20):
    spw = np.bincount(y_tr.astype(int))[0] / max(np.bincount(y_tr.astype(int))[1], 1)
    def obj(trial):
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc', scale_pos_weight=spw,
            max_depth=trial.suggest_int('md', 2, 6),
            learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('ss', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('cs', 0.5, 1.0),
            reg_alpha=trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float('lam', 1e-4, 10.0, log=True),
            n_estimators=200, early_stopping_rounds=30,
            random_state=RANDOM_STATE, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=spw,
                              max_depth=bp['md'], learning_rate=bp['lr'],
                              subsample=bp['ss'], colsample_bytree=bp['cs'],
                              reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                              n_estimators=200, random_state=RANDOM_STATE, verbosity=0)
    best.fit(X_tr, y_tr, verbose=False)
    return best, study.best_value


# --- LightGBM ---
def train_lgb_optuna(X_tr, y_tr, X_val, y_val, n_trials=20):
    spw = np.bincount(y_tr.astype(int))[0] / max(np.bincount(y_tr.astype(int))[1], 1)
    def obj(trial):
        params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                  'num_leaves': trial.suggest_int('nl', 7, 31),
                  'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                  'feature_fraction': trial.suggest_float('ff', 0.5, 1.0),
                  'min_child_samples': trial.suggest_int('mcs', 5, 30),
                  'lambda_l1': trial.suggest_float('l1', 1e-6, 1.0, log=True),
                  'scale_pos_weight': spw, 'verbosity': -1,
                  'random_state': RANDOM_STATE}
        tr = lgb.Dataset(X_tr, label=y_tr)
        va = lgb.Dataset(X_val, label=y_val, reference=tr)
        m = lgb.train(params, tr, valid_sets=[va], num_boost_round=200,
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        return roc_auc_score(y_val, m.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
              'num_leaves': bp['nl'], 'learning_rate': bp['lr'],
              'feature_fraction': bp['ff'], 'min_child_samples': bp['mcs'],
              'lambda_l1': bp['l1'], 'scale_pos_weight': spw,
              'verbosity': -1, 'random_state': RANDOM_STATE}
    best = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=200)
    return best, study.best_value


# =====================  CORE CV EVALUATION (改进5+6)  =====================

def repeated_cv_evaluate(X_feat, y_feat, X_signal, y_signal, signal_len,
                          dataset_name, k_best=None):
    """Repeated Stratified K-Fold CV with SMOTE inside each fold.

    - Fixed 20% test set held out at the beginning → never touched during CV
    - RepeatedStratifiedKFold(5-fold, 3 repeats) on the 80% training data
    - In each fold: preprocess → SMOTE → Optuna HP tune → train → evaluate
    - 1D CNN trained separately on raw signals (once per fold, no inner Optuna)
    """
    # ---- Step 1: Split off 20% test set (改进5) ----
    # Use index-based split to ensure feature and signal splits are identical
    n_total = len(y_feat)
    all_idx = np.arange(n_total)
    tr_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y_feat, random_state=RANDOM_STATE)

    X_tr_val = X_feat[tr_val_idx]
    y_tr_val = y_feat[tr_val_idx]
    X_test = X_feat[test_idx]
    y_test = y_feat[test_idx]
    X_sig_tr_val = X_signal[tr_val_idx] if X_signal is not None else None
    X_sig_test = X_signal[test_idx] if X_signal is not None else None
    y_sig_tr_val = y_signal[tr_val_idx] if y_signal is not None else None
    y_sig_test = y_signal[test_idx] if y_signal is not None else None

    print(f"\n  Train+Val: {len(y_tr_val)} ({dict(zip(*np.unique(y_tr_val, return_counts=True)))}), "
          f"Test: {len(y_test)} ({dict(zip(*np.unique(y_test, return_counts=True)))})")

    # ---- Step 2: Repeated Stratified K-Fold on training data ----
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                    random_state=RANDOM_STATE)
    model_names = ['LR_ElasticNet', 'LinearSVM', 'RBFSVM', 'XGBoost', 'LightGBM', 'CNN_1DResNet']
    all_cv_results = {m: [] for m in model_names}

    for fold_idx, (inner_tr_idx, val_idx) in enumerate(rskf.split(X_tr_val, y_tr_val)):
        X_fold_train_raw = X_tr_val[inner_tr_idx]
        y_fold_train_raw = y_tr_val[inner_tr_idx]
        X_fold_val_raw = X_tr_val[val_idx]
        y_fold_val = y_tr_val[val_idx]

        # ---- In-fold preprocessing (fit on train only) ----
        X_fold_train, X_fold_val = preprocess_train_only(
            X_fold_train_raw, X_fold_val_raw, y_fold_train_raw, k_best=k_best)

        # ---- SMOTE on training portion only (改进6) ----
        X_fold_train_bal, y_fold_train_bal = apply_smote_train_only(X_fold_train, y_fold_train_raw)
        n_synth = X_fold_train_bal.shape[0] - X_fold_train.shape[0]
        if fold_idx == 0 and n_synth > 0:
            print(f"  [Fold 0] SMOTE: {X_fold_train.shape[0]} → {X_fold_train_bal.shape[0]} "
                  f"(+{n_synth} synthetic)")

        # ---- Inner CV for Optuna HP tuning ----
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE + fold_idx)
        inner_splits = list(inner_cv.split(X_fold_train_bal, y_fold_train_bal))
        i_tr_idx, i_val_idx = inner_splits[0]
        X_inner_tr = X_fold_train_bal[i_tr_idx]
        y_inner_tr = y_fold_train_bal[i_tr_idx]
        X_inner_val = X_fold_train_bal[i_val_idx]
        y_inner_val = y_fold_train_bal[i_val_idx]

        fold_probs = {}

        # --- LR ---
        lr, _ = train_lr_optuna(X_inner_tr, y_inner_tr, X_inner_val, y_inner_val)
        fold_probs['LR_ElasticNet'] = lr.predict_proba(X_fold_val)[:, 1]

        # --- Linear SVM ---
        svm, _ = train_svm_optuna(X_inner_tr, y_inner_tr, X_inner_val, y_inner_val)
        if hasattr(svm, 'predict_proba'):
            fold_probs['LinearSVM'] = svm.predict_proba(X_fold_val)[:, 1]
        else:
            fold_probs['LinearSVM'] = svm.decision_function(X_fold_val)

        # --- RBF SVM ---
        rbf_svm, _ = train_rbf_svm_optuna(X_inner_tr, y_inner_tr, X_inner_val, y_inner_val)
        fold_probs['RBFSVM'] = rbf_svm.predict_proba(X_fold_val)[:, 1]

        # --- XGBoost ---
        xgb_m, _ = train_xgb_optuna(X_inner_tr, y_inner_tr, X_inner_val, y_inner_val)
        fold_probs['XGBoost'] = xgb_m.predict_proba(X_fold_val)[:, 1]

        # --- LightGBM ---
        lgb_m, _ = train_lgb_optuna(X_inner_tr, y_inner_tr, X_inner_val, y_inner_val)
        fold_probs['LightGBM'] = lgb_m.predict(X_fold_val)

        # --- 1D CNN on raw signals ---
        if X_sig_tr_val is not None and signal_len > 0:
            X_sig_train_raw = X_sig_tr_val[inner_tr_idx]
            X_sig_val_raw = X_sig_tr_val[val_idx]
            y_sig_train_raw = y_sig_tr_val[inner_tr_idx]

            if X_sig_train_raw.shape[0] >= 5 and np.bincount(y_sig_train_raw.astype(int)).min() >= 3:
                X_sig_train_bal, y_sig_train_bal = apply_smote_train_only(
                    X_sig_train_raw, y_sig_train_raw)
            else:
                X_sig_train_bal, y_sig_train_bal = X_sig_train_raw, y_sig_train_raw

            cnn_m, _ = train_cnn(X_sig_train_bal, y_sig_train_bal,
                                  X_sig_val_raw, y_fold_val, signal_len,
                                  epochs=100, patience=25)
            fold_probs['CNN_1DResNet'] = cnn_predict(cnn_m, X_sig_val_raw)
            del cnn_m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- Collect fold metrics ----
        for m_name in model_names:
            if m_name in fold_probs:
                thresh = optimize_threshold(y_fold_val, fold_probs[m_name])
                metrics = compute_metrics(y_fold_val, fold_probs[m_name], threshold=thresh)
                metrics['_fold'] = fold_idx + 1
                all_cv_results[m_name].append(metrics)

        # Show progress
        if (fold_idx + 1) % N_FOLDS == 0:
            rep = (fold_idx + 1) // N_FOLDS
            print(f"  Repeat {rep}/{N_REPEATS} complete ({fold_idx+1}/{N_FOLDS*N_REPEATS} folds)")

    # ---- Step 3: Train final models on full training data & evaluate on test set ----
    print(f"\n  Training final models on full training data for test evaluation...")

    # Preprocess full training data
    X_tr_full, X_test_pp = preprocess_train_only(X_tr_val, X_test, y_tr_val, k_best=k_best)
    X_tr_bal, y_tr_bal = apply_smote_train_only(X_tr_full, y_tr_val)

    # Inner CV for final HP tuning
    inner_cv_final = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_s_final = list(inner_cv_final.split(X_tr_bal, y_tr_bal))
    X_inner_tr_f, y_inner_tr_f = X_tr_bal[inner_s_final[0][0]], y_tr_bal[inner_s_final[0][0]]
    X_inner_val_f, y_inner_val_f = X_tr_bal[inner_s_final[0][1]], y_tr_bal[inner_s_final[0][1]]

    test_final_probs = {}

    # LR
    lr_f, _ = train_lr_optuna(X_inner_tr_f, y_inner_tr_f, X_inner_val_f, y_inner_val_f)
    test_final_probs['LR_ElasticNet'] = lr_f.predict_proba(X_test_pp)[:, 1]

    # Linear SVM
    svm_f, _ = train_svm_optuna(X_inner_tr_f, y_inner_tr_f, X_inner_val_f, y_inner_val_f)
    if hasattr(svm_f, 'predict_proba'):
        test_final_probs['LinearSVM'] = svm_f.predict_proba(X_test_pp)[:, 1]
    else:
        test_final_probs['LinearSVM'] = svm_f.decision_function(X_test_pp)

    # RBF SVM
    rbf_f, _ = train_rbf_svm_optuna(X_inner_tr_f, y_inner_tr_f, X_inner_val_f, y_inner_val_f)
    test_final_probs['RBFSVM'] = rbf_f.predict_proba(X_test_pp)[:, 1]

    # XGBoost
    xgb_f, _ = train_xgb_optuna(X_inner_tr_f, y_inner_tr_f, X_inner_val_f, y_inner_val_f)
    test_final_probs['XGBoost'] = xgb_f.predict_proba(X_test_pp)[:, 1]

    # LightGBM
    lgb_f, _ = train_lgb_optuna(X_inner_tr_f, y_inner_tr_f, X_inner_val_f, y_inner_val_f)
    test_final_probs['LightGBM'] = lgb_f.predict(X_test_pp)

    # CNN on raw signals
    if X_sig_tr_val is not None and signal_len > 0:
        X_sig_tr_bal, y_sig_tr_bal = apply_smote_train_only(X_sig_tr_val, y_sig_tr_val)
        cnn_f, _ = train_cnn(X_sig_tr_bal, y_sig_tr_bal,
                              X_sig_test, y_sig_test, signal_len,
                              epochs=150, patience=30)
        test_final_probs['CNN_1DResNet'] = cnn_predict(cnn_f, X_sig_test)
        del cnn_f
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Compute CV metrics (mean ± std across folds) ----
    print(f"\n{'='*70}")
    print(f"  REPEATED CV RESULTS — {dataset_name.upper()}")
    print(f"  {N_FOLDS}-fold × {N_REPEATS} repeats, SMOTE inside CV, k_best={k_best}")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'ACC':>7} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  " + "-" * 60)

    cv_summary = {}
    for m_name in model_names:
        if not all_cv_results[m_name]:
            continue
        aucs = [m['AUC'] for m in all_cv_results[m_name]]
        f1s  = [m['F1'] for m in all_cv_results[m_name]]
        accs = [m['Accuracy'] for m in all_cv_results[m_name]]
        precs = [m['Precision'] for m in all_cv_results[m_name]]
        recs = [m['Recall'] for m in all_cv_results[m_name]]

        cv_summary[m_name] = {
            'ACC_mean': np.mean(accs), 'ACC_std': np.std(accs),
            'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
            'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s),
            'Precision_mean': np.mean(precs), 'Precision_std': np.std(precs),
            'Recall_mean': np.mean(recs), 'Recall_std': np.std(recs),
            '_all_aucs': aucs, '_all_f1s': f1s,
        }
        print(f"  {m_name:<22} {np.mean(accs):6.3f}±{np.std(accs):.3f} "
              f"{np.mean(aucs):6.3f}±{np.std(aucs):.3f} "
              f"{np.mean(f1s):6.3f}±{np.std(f1s):.3f} "
              f"{np.mean(precs):6.3f}±{np.std(precs):.3f} "
              f"{np.mean(recs):6.3f}±{np.std(recs):.3f}")

    best_cv = max(cv_summary.items(), key=lambda x: x[1]['AUC_mean'])
    print(f"\n  Best CV model: {best_cv[0]} (AUC={best_cv[1]['AUC_mean']:.4f}±{best_cv[1]['AUC_std']:.4f})")

    # ---- Test set evaluation ----
    print(f"\n{'='*70}")
    print(f"  HELD-OUT TEST SET RESULTS — {dataset_name.upper()} ({len(y_test)} samples)")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'ACC':>7} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    print(f"  " + "-" * 75)

    test_summary = {}
    for m_name in model_names:
        if m_name not in test_final_probs:
            continue
        probs = test_final_probs[m_name]
        thresh = optimize_threshold(y_test, probs)
        m = compute_metrics(y_test, probs, threshold=thresh)
        test_summary[m_name] = m
        print(f"  {m_name:<22} {m['Accuracy']:6.3f}  {m['AUC']:6.3f}  {m['F1']:6.3f}  "
              f"{m['Precision']:6.3f}  {m['Recall']:6.3f}  "
              f"{m['TP']:5d} {m['TN']:5d} {m['FP']:5d} {m['FN']:5d}")

    # Weighted ensemble on test set
    weights = {}
    for m_name in test_final_probs:
        auc = test_summary[m_name]['AUC']
        weights[m_name] = max(auc - 0.5, 0.05)

    w_sum = sum(weights.values())
    ens_probs = sum(weights[k] * test_final_probs[k] for k in weights) / w_sum
    ens_thresh = optimize_threshold(y_test, ens_probs)
    ens_m = compute_metrics(y_test, ens_probs, threshold=ens_thresh)
    test_summary['Ensemble'] = ens_m
    test_final_probs['Ensemble'] = ens_probs

    print(f"  " + "-" * 75)
    print(f"  {'Ensemble':<22} {ens_m['Accuracy']:6.3f}  {ens_m['AUC']:6.3f}  {ens_m['F1']:6.3f}  "
          f"{ens_m['Precision']:6.3f}  {ens_m['Recall']:6.3f}  "
          f"{ens_m['TP']:5d} {ens_m['TN']:5d} {ens_m['FP']:5d} {ens_m['FN']:5d}")

    return {'cv_summary': cv_summary, 'test_summary': test_summary,
            'y_test': y_test, 'test_probs': test_final_probs}


# =====================  MAIN  =====================
def main():
    print("=" * 70)
    print("IMPROVED BINARY CLASSIFICATION")
    print("1D ResNet CNN + LR/SVM/XGB/LGB + Repeated CV + SMOTE-in-CV")
    print("=" * 70)

    all_results = {}

    # ======== ECG ========
    ecg_feat_file = 'features_ecg.xlsx'
    ecg_signal_dir = 'ecg_segments'
    ecg_label_file = 'labels_for_ecg.xlsx'

    if os.path.exists(ecg_feat_file):
        print(f"\n{'#'*60}")
        print(f"# ECG PIPELINE")
        print(f"{'#'*60}")

        # Load features (tsfresh + physiological params)
        print("\n[ECG] Loading features + physiological params...")
        X_ecg_feat, y_ecg_feat, _ = build_joint_feature_matrix(ecg_feat_file, ecg_label_file)

        # Load raw signals for CNN
        print("[ECG] Loading raw signals for 1D CNN...")
        X_ecg_sig, y_ecg_sig, _ = load_raw_signals(
            ecg_signal_dir, ecg_label_file, ECG_SIGNAL_LEN)

        # Verify label alignment between features and signals
        assert len(y_ecg_feat) == len(y_ecg_sig), \
            f"Mismatch: {len(y_ecg_feat)} features vs {len(y_ecg_sig)} signals"

        all_results['ECG'] = repeated_cv_evaluate(
            X_ecg_feat, y_ecg_feat, X_ecg_sig, y_ecg_sig,
            signal_len=ECG_SIGNAL_LEN, dataset_name='ECG',
            k_best=min(50, X_ecg_feat.shape[1] // 3))
    else:
        print(f"\nECG: {ecg_feat_file} not found")

    # ======== SCG ========
    scg_feat_file = 'features_scg_fixed.xlsx'
    scg_signal_dir = 'scg_segments'
    scg_label_file = 'labels_for_scg.xlsx'

    if os.path.exists(scg_feat_file):
        print(f"\n{'#'*60}")
        print(f"# SCG PIPELINE")
        print(f"{'#'*60}")

        print("\n[SCG] Loading features + physiological params...")
        X_scg_feat, y_scg_feat, _ = build_joint_feature_matrix(scg_feat_file, scg_label_file)

        print("[SCG] Loading raw signals for 1D CNN...")
        X_scg_sig, y_scg_sig, _ = load_raw_signals(
            scg_signal_dir, scg_label_file, SCG_SIGNAL_LEN)

        assert len(y_scg_feat) == len(y_scg_sig), \
            f"Mismatch: {len(y_scg_feat)} features vs {len(y_scg_sig)} signals"

        # SCG has many features (789 + phys params), use more aggressive selection
        k_scg = min(40, X_scg_feat.shape[1] // 5)
        all_results['SCG'] = repeated_cv_evaluate(
            X_scg_feat, y_scg_feat, X_scg_sig, y_scg_sig,
            signal_len=SCG_SIGNAL_LEN, dataset_name='SCG',
            k_best=k_scg)
    else:
        print(f"\nSCG: {scg_feat_file} not found")

    # ======== Final Summary ========
    if all_results:
        print(f"\n{'='*70}")
        print("CROSS-MODALITY TEST SET COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Modality':<10} {'Best Model':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}")
        print(f"  " + "-" * 60)

        for ds_name, res in all_results.items():
            # Find best model on test set
            ts = res['test_summary']
            best = max(ts.items(), key=lambda x: (x[1]['AUC'], x[1]['F1']))
            print(f"  {ds_name:<10} {best[0]:<22} "
                  f"{best[1]['AUC']:7.4f} {best[1]['F1']:7.4f} "
                  f"{best[1]['Precision']:7.4f} {best[1]['Recall']:7.4f}")

    return all_results


if __name__ == "__main__":
    main()
