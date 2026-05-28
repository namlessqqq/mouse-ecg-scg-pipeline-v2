"""
Step 2: Rebuild labels with EF < 50 threshold
=============================================
For each segment file in ecg_segments/ and scg_segments/:
  1. Parse mouse_id and date from filename
  2. Find EF value: labels.xlsx first, then TAC Excel files by date
  3. Label: EF < 50 → 1 (heart failure), EF >= 50 → 0 (normal)
  4. Output: labels_for_ecg.xlsx, labels_for_scg.xlsx
"""
import os, re, sys
import numpy as np
import pandas as pd

# ---- CONFIG ----
LABELS_XLSX = 'temp_data/labels.xlsx'
TAC_FILES = {
    '4.29': 'temp_data/20250429 TTT TAC 数据处理.xls',
    '4.30': 'temp_data/20250429 TTT TAC 数据处理.xls',
    '5.14': 'temp_data/20250515 TAC.xlsx',
    '5.28': 'temp_data/20250528 TTT TAC数据处理.xlsx',
    '6.11': 'temp_data/20250611 TTT TAC数据处理.xlsx',
    '6.25': 'temp_data/20250625 TTT TAC 数据处理.xlsx',
    '7.10': 'temp_data/20250710 TTT TAC 数据处理.xlsx',
    '4.18': None,  # Only 1 mouse (T366), will be found in labels.xlsx
}

EF_THRESHOLD = 50.0
ECG_SEG_DIR = 'ecg_segments'
SCG_SEG_DIR = 'scg_segments'


def clean_mouse_id(raw_name):
    """Same cleaning logic as clean_raw_data.py."""
    name = str(raw_name).strip()
    if name.lower().endswith('.csv'):
        name = name[:-4]
    name = re.sub(r'[（(][^)）]*[)）]', '', name)
    name = re.sub(r'^[Tt][-−](\d)', r'T\1', name)
    name = re.sub(r'[-−]\d+$', '', name)
    name = re.sub(r'[-−].*$', '', name)
    name = re.sub(r'[一-鿿㐀-䶿豈-﫿　-〿＀-￯]', '', name)
    name = re.sub(r'[^\x00-\x7f]', '', name)
    name = re.sub(r'[^a-zA-Z0-9]', '', name)
    return name.strip().upper()


def load_labels_xlsx_ef_map():
    """Build {cleaned_mouse_id: EF_value} from labels.xlsx."""
    df = pd.read_excel(LABELS_XLSX)
    ef_map = {}
    for _, row in df.iterrows():
        ori = str(row['ori_name'])
        mid = clean_mouse_id(ori)
        ef = row['Ejection Fraction']
        if mid and not pd.isna(ef):
            # If multiple entries for same mouse, keep the first (or average)
            if mid not in ef_map:
                ef_map[mid] = float(ef)
    print(f"  labels.xlsx: {len(ef_map)} unique mice with EF values")
    return ef_map


def load_tac_ef_map(tac_path):
    """Build {cleaned_mouse_id: EF_value} from a TAC Excel file."""
    if tac_path is None or not os.path.exists(tac_path):
        return {}

    try:
        df = pd.read_excel(tac_path, engine='openpyxl')
    except Exception:
        try:
            df = pd.read_excel(tac_path)  # fallback to default engine
        except Exception as e:
            print(f"    WARN: Cannot read {tac_path}: {e}")
            return {}

    # TAC files have different formats. Try to find mouse ID and EF columns.
    # Common pattern: Unnamed:0 = group, Unnamed:1 = mouse_id, then EF column
    id_col = None
    ef_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'ejection fraction' in col_lower or col_lower.strip() == 'ejection fraction/%':
            ef_col = col
        if 'unnamed:1' in col_lower or 'ori_name' in col_lower or 'subject' in col_lower:
            id_col = col

    # For older .xls format (20250429): different structure
    if id_col is None:
        # Try first two unnamed columns
        unnamed_cols = [c for c in df.columns if 'unnamed' in str(c).lower()]
        if len(unnamed_cols) >= 2:
            id_col = unnamed_cols[1]  # Second unnamed is usually mouse ID

    # If still not found, try column with '鼠' in name or first text column
    if id_col is None:
        for col in df.columns:
            vals = df[col].dropna().astype(str)
            # Look for column with T-prefixed values
            if any(v.strip().upper().startswith('T') for v in vals.head(20)):
                id_col = col
                break

    if id_col is None or ef_col is None:
        print(f"    WARN: Could not find id_col or ef_col in {tac_path}")
        print(f"    Columns: {list(df.columns)}")
        return {}

    ef_map = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        mid = clean_mouse_id(raw_id)
        ef = row[ef_col]
        if mid and not pd.isna(ef):
            ef_val = float(ef)
            if 20 < ef_val < 100:  # Sanity check: EF must be in reasonable range
                if mid not in ef_map:
                    ef_map[mid] = ef_val
    return ef_map


def build_ef_lookup():
    """Build complete EF lookup: labels.xlsx + all TAC files by date.

    Returns {date: {mouse_id: EF_value}}
    """
    # Base: labels.xlsx (global, no date)
    global_ef = load_labels_xlsx_ef_map()

    # Date-specific: TAC files
    date_ef = {}
    for date_dir, tac_path in TAC_FILES.items():
        if tac_path and os.path.exists(tac_path):
            tac_map = load_tac_ef_map(tac_path)
            date_ef[date_dir] = tac_map
            print(f"  TAC {date_dir}: {len(tac_map)} mice")
        else:
            date_ef[date_dir] = {}

    return global_ef, date_ef


def get_ef_for_segment(mouse_id, date_dir, global_ef, date_ef):
    """Get EF value for a segment.

    Priority: date-specific TAC file > global labels.xlsx
    """
    # Try date-specific TAC first
    if date_dir in date_ef and mouse_id in date_ef[date_dir]:
        return date_ef[date_dir][mouse_id]

    # Fall back to global labels.xlsx
    if mouse_id in global_ef:
        return global_ef[mouse_id]

    # Try case-insensitive match
    mid_upper = mouse_id.upper()
    for mid, ef in global_ef.items():
        if mid.upper() == mid_upper:
            return ef

    if date_dir in date_ef:
        for mid, ef in date_ef[date_dir].items():
            if mid.upper() == mid_upper:
                return ef

    return None


def build_labels():
    """Build label files for ECG and SCG segments."""
    global_ef, date_ef = build_ef_lookup()

    for modality, seg_dir in [('ECG', ECG_SEG_DIR), ('SCG', SCG_SEG_DIR)]:
        if not os.path.isdir(seg_dir):
            print(f"  {modality}: {seg_dir} not found, skipping")
            continue

        files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.csv')])
        print(f"\n  {modality}: {len(files)} segments")

        labels = []
        missing_ef = 0

        for fname in files:
            # Parse: {month}_{day}_{mouse_id}_seg{idx}.csv
            # e.g., 4_18_T366_seg0.csv, 5_28_T356_seg14.csv
            base = fname.replace('.csv', '')

            if '_seg' not in base:
                continue

            prefix, seg_num = base.rsplit('_seg', 1)
            parts = prefix.split('_')
            if len(parts) < 3:
                continue

            date_prefix = f"{parts[0]}.{parts[1]}"
            mouse_id = '_'.join(parts[2:])

            # Get EF for this mouse/date
            ef = get_ef_for_segment(mouse_id, date_prefix, global_ef, date_ef)

            if ef is None:
                missing_ef += 1
                if missing_ef <= 5:
                    print(f"    WARN: No EF for mouse={mouse_id}, date={date_prefix}")
                continue

            label = 1 if ef < EF_THRESHOLD else 0
            labels.append({
                'filename': fname,
                'label': label,
                'EF': ef,
            })

        if missing_ef > 0:
            print(f"    Missing EF: {missing_ef}/{len(files)} segments skipped")

        # Build DataFrame and save
        df_out = pd.DataFrame(labels)
        # Add placeholder columns for compatibility with downstream code
        # (feature_extract.py expects label in last column)
        out_path = f'labels_for_{modality.lower()}.xlsx'

        # Simple format: filename, label
        df_simple = df_out[['filename', 'label']].copy()
        df_simple.to_excel(out_path, index=False)

        # Statistics
        class_counts = df_out['label'].value_counts().to_dict()
        print(f"    Output: {out_path} ({len(df_out)} rows)")
        print(f"    Class distribution: {class_counts}")
        if 1 in class_counts and 0 in class_counts:
            ratio = class_counts[0] / class_counts[1]
            print(f"    Imbalance ratio: {ratio:.1f}:1")


if __name__ == '__main__':
    build_labels()
