"""
Step 0: Label filtering + filename cleanup
===========================================
- Delete raw CSV files that have no corresponding label in labels.xlsx
- Remove Chinese characters and quality annotations from filenames
- Deduplicate: keep the best-quality version per mouse per date
- Normalize to {mouse_id}.csv format
"""
import os, shutil, re
import pandas as pd
from collections import defaultdict

TEMP_DATA = 'temp_data'
LABELS_FILE = 'temp_data/labels.xlsx'
RAW_DATA_DIR = 'temp_data/raw_clean'  # Output directory for cleaned files

# Date folders
DATE_DIRS = ['4.18', '4.29', '4.30', '5.14', '5.28', '6.11', '6.25', '7.10']


def load_valid_ids():
    """Load all ori_name that have labels from labels.xlsx."""
    df = pd.read_excel(LABELS_FILE)
    valid = set()
    for _, row in df.iterrows():
        ori = str(row['ori_name']).strip()
        # Normalize: remove Chinese, parenthesized content, hyphens, uppercase
        clean = clean_mouse_id(ori)
        valid.add(clean)
    print(f"Loaded {len(valid)} valid mouse IDs from labels.xlsx")
    return valid


def clean_mouse_id(raw_name):
    """Extract and normalize a mouse ID from a raw filename or ori_name.

    Examples:
      'T366' -> 'T366'
      't372（0）' -> 'T372'
      't374心衰（ECG0）' -> 'T374'
      'T-366' -> 'T366'
      'T427-心衰' -> 'T427'
      'T517好）' -> 'T517'
      'T511-信号不太好' -> 'T511'
      'T481(信号不太好)' -> 'T481'
      'T436(2)' -> 'T436'
      'T422-2(0)' -> 'T422'
      '458' -> '458'
      't508(2)' -> 'T508'
      'Y101' -> 'Y101'
    """
    name = str(raw_name).strip()
    if name.lower().endswith('.csv'):
        name = name[:-4]

    # Step 1: Remove parenthesized content FIRST
    name = re.sub(r'[（(][^)）]*[)）]', '', name)

    # Step 2: Normalize T-NNN format: T-366 -> T366 (hyphen between letter and number)
    name = re.sub(r'^[Tt][-−](\d)', r'T\1', name)

    # Step 3: Remove trailing replicate numbers: T422-2 -> T422
    name = re.sub(r'[-−]\d+$', '', name)

    # Step 4: Remove remaining hyphen-separated annotations (Chinese/alpha after hyphen)
    name = re.sub(r'[-−].*$', '', name)

    # Step 5: Remove Chinese characters
    name = re.sub(r'[一-鿿㐀-䶿豈-﫿　-〿＀-￯]', '', name)
    name = re.sub(r'[^\x00-\x7f]', '', name)

    # Step 6: Clean up and normalize case
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9]', '', name)
    name = name.upper()

    return name if name else ''


def quality_score(filename):
    """Score a filename by quality (higher = better).

    Priority: clean name > English annotation > Chinese annotation > has (0) marker
    """
    score = 10
    # Penalize Chinese characters
    if re.search(r'[一-鿿]', filename):
        score -= 5
    # Penalize (0) or (2) markers (often indicate problematic channels)
    if re.search(r'[（(]\d[)）]', filename):
        score -= 3
    # Penalize hyphen annotations
    if re.search(r'[-−]', filename):
        score -= 2
    # Penalize '差' (poor), '不好' (bad), '问题' (problem) in original
    if any(kw in filename for kw in ['差', '不好', '问题', '去掉', '去除']):
        score -= 4
    return score


def deduplicate_files(file_list):
    """Given a list of (filename, filepath) for the same mouse in same date,
    return only the best one."""
    if len(file_list) <= 1:
        return file_list
    # Sort by quality score descending, take best
    scored = [(quality_score(fn), fn, fp) for fn, fp in file_list]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    if len(file_list) > 1:
        print(f"    Dedup: keeping '{best[1]}' (score={best[0]}), "
              f"dropping {[s[1] for s in scored[1:]]}")
    return [(best[1], best[2])]


def process(valid_ids):
    """Main processing: filter, clean, deduplicate."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    total_kept = 0
    total_deleted = 0

    # Group files by (date, mouse_id)
    file_groups = defaultdict(list)

    for date_dir in DATE_DIRS:
        src_dir = os.path.join(TEMP_DATA, date_dir)
        if not os.path.isdir(src_dir):
            continue

        out_dir = os.path.join(RAW_DATA_DIR, date_dir)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            if not fname.lower().endswith('.csv'):
                continue

            src_path = os.path.join(src_dir, fname)
            mouse_id = clean_mouse_id(fname)

            if not mouse_id:
                print(f"  WARN: Could not extract mouse ID from '{fname}'")
                continue

            if mouse_id not in valid_ids:
                print(f"  DELETE (no label): {date_dir}/{fname}")
                total_deleted += 1
                continue

            file_groups[(date_dir, mouse_id)].append((fname, src_path))

    # Process each group: deduplicate and copy
    for (date_dir, mouse_id), files in file_groups.items():
        out_dir = os.path.join(RAW_DATA_DIR, date_dir)
        best_files = deduplicate_files(files)

        for orig_fname, src_path in best_files:
            dst_name = f"{mouse_id}.csv"
            dst_path = os.path.join(out_dir, dst_name)

            # Handle case where dedup results in same output name
            if os.path.exists(dst_path):
                base = mouse_id
                idx = 1
                while os.path.exists(os.path.join(out_dir, f"{base}_{idx}.csv")):
                    idx += 1
                dst_name = f"{base}_{idx}.csv"
                dst_path = os.path.join(out_dir, dst_name)

            shutil.copy2(src_path, dst_path)
            total_kept += 1
            if orig_fname != dst_name:
                print(f"  COPY: {date_dir}/{orig_fname} -> {date_dir}/{dst_name}")

    print(f"\nDone: {total_kept} files kept, {total_deleted} deleted (no label)")
    print(f"Output: {RAW_DATA_DIR}/")

    # Summary per date
    for date_dir in DATE_DIRS:
        out_dir = os.path.join(RAW_DATA_DIR, date_dir)
        if os.path.isdir(out_dir):
            count = len([f for f in os.listdir(out_dir) if f.endswith('.csv')])
            print(f"  {date_dir}: {count} files")


if __name__ == '__main__':
    valid_ids = load_valid_ids()
    process(valid_ids)
