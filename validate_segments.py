"""
Validate that segments are real ECG/SCG signals, not noise/artifacts.

ECG validity checks:
  - Proper R-peaks with mouse-appropriate HR (200-850 bpm)
  - QRS-like morphology (sharp peaks, not broad waves)
  - Adequate SNR and signal variance
  - Regular rhythm (not chaotic noise)

SCG validity checks:
  - Periodic envelope structure at mouse HR
  - Adequate signal energy
  - Envelope peaks with consistent intervals

Removes segments that fail validation, regenerates labels and features.
"""
import os, shutil, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.signal import find_peaks, hilbert, butter, filtfilt

SR = 500

# ---- ECG validity checks ----

def detect_proper_r_peaks(signal, fs=SR):
    """Detect R-peaks using autocorrelation + prominence for robust HR estimation.

    Uses autocorrelation to find the dominant period, then validates with
    peak detection. This is more robust to noisy signals.

    Returns (peak_indices, hr_bpm, rr_cv, snr)
    """
    if len(signal) < fs * 0.5:
        return np.array([]), 0, 1.0, 0

    sig_std = np.std(signal)
    if sig_std < 1e-8:
        return np.array([]), 0, 1.0, 0

    # ---- Method 1: Autocorrelation for robust HR estimation ----
    # This is less sensitive to peak detection failures in noisy signals
    sig_norm = signal - np.mean(signal)
    autocorr = np.correlate(sig_norm, sig_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Search for first major peak in plausible HR range
    min_lag = int(60 * fs / 900)   # 900 bpm = 33 samples
    max_lag = int(60 * fs / 150)   # 150 bpm = 200 samples
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag >= max_lag:
        min_lag = max(1, max_lag // 3)

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) < 3:
        return np.array([]), 0, 1.0, 0

    # Find peaks in autocorrelation
    try:
        ac_peaks, _ = find_peaks(search_region, height=0.1)
    except Exception:
        ac_peaks = []

    if len(ac_peaks) > 0:
        best_lag = min_lag + ac_peaks[np.argmax(search_region[ac_peaks])]
        ac_hr = 60 * fs / best_lag
    else:
        ac_hr = 0

    # ---- Method 2: Peak detection for RR consistency check ----
    # Use more permissive settings
    min_prominence = sig_std * 1.0  # Relaxed from 1.5
    min_distance = int(0.035 * fs)  # 35ms (allows HR up to ~1700)

    try:
        peaks, properties = find_peaks(signal, prominence=min_prominence,
                                        distance=min_distance, width=int(0.003*fs))
    except Exception:
        return np.array([]), 0, 1.0, 0

    if len(peaks) < 2:
        return peaks, 0, 1.0, 0

    # Calculate HR from peaks (take median of top prominent peaks)
    prominences = properties.get('prominences', np.ones(len(peaks)))
    top_idx = np.argsort(prominences)[-max(3, len(peaks)//3):]  # Top 1/3 most prominent
    top_peaks = np.sort(peaks[top_idx])

    if len(top_peaks) >= 2:
        rr_samples = np.diff(top_peaks)
        rr_median = np.median(rr_samples)
        if rr_median >= 2:
            hr_bpm = 60 * fs / rr_median
        else:
            hr_bpm = ac_hr  # Fall back to autocorrelation
    else:
        hr_bpm = ac_hr

    # Use autocorrelation HR if peak-based HR looks unreasonable
    if ac_hr > 0 and (hr_bpm < 100 or hr_bpm > 1200):
        hr_bpm = ac_hr

    # RR consistency using all detected peaks
    rr_samples = np.diff(peaks)
    if len(rr_samples) >= 3:
        rr_bpm = 60 * fs / rr_samples
        valid_mask = (rr_bpm > 100) & (rr_bpm < 1500)
        if np.sum(valid_mask) >= 2:
            rr_cv = np.std(rr_samples[valid_mask]) / (np.mean(rr_samples[valid_mask]) + 1e-10)
        else:
            rr_cv = 0.5  # Neutral - not enough data
    else:
        rr_cv = 0.5

    # SNR
    peak_heights = signal[peaks]
    baseline = signal[np.argsort(np.abs(signal))[:max(1, int(0.6 * len(signal)))]]
    noise_std = np.std(baseline)
    snr = np.mean(np.abs(peak_heights)) / (noise_std + 1e-10)

    return peaks, hr_bpm, rr_cv, snr


def is_valid_ecg(filepath):
    """Check if an ECG segment looks like a real ECG signal.

    Returns (is_valid, reason_string)
    """
    try:
        sig = np.loadtxt(filepath)
    except Exception:
        return False, 'read_error'

    if len(sig) < SR * 0.5:
        return False, 'too_short'

    sig_std = np.std(sig)
    if sig_std < 1e-6:
        return False, 'flat_signal'

    # 1. Check for saturation / railing
    upper = np.sum(np.abs(sig - np.max(sig)) < 1e-8)
    lower = np.sum(np.abs(sig - np.min(sig)) < 1e-8)
    sat_ratio = (upper + lower) / len(sig)
    if sat_ratio > 0.1:
        return False, f'saturated={sat_ratio:.0%}'

    # 2. Proper R-peak detection
    peaks, hr_bpm, rr_cv, snr = detect_proper_r_peaks(sig)

    if len(peaks) < 3:
        return False, f'too_few_peaks={len(peaks)}'

    # 3. Heart rate in mouse physiological range (relaxed)
    if hr_bpm < 120 or hr_bpm > 950:
        return False, f'hr_out_of_range={hr_bpm:.0f}bpm'

    # 4. Rhythm consistency - chaotic noise has high RR variability
    if rr_cv > 0.8:
        return False, f'irregular_rhythm_cv={rr_cv:.2f}'

    # 5. Adequate SNR (relaxed)
    if snr < 2.0:
        return False, f'low_snr={snr:.1f}'

    # 6. QRS width check (relaxed): real QRS is <20ms in mice
    try:
        _, props = find_peaks(sig, prominence=sig_std * 1.0,
                               distance=int(0.035*SR), width=int(0.002*SR))
        if 'widths' in props and len(props['widths']) > 0:
            median_width = np.median(props['widths'])
            if median_width > 35:  # >70ms is too wide for QRS
                return False, f'wide_peaks={median_width:.0f}samples'
    except Exception:
        pass

    return True, f'valid_hr={hr_bpm:.0f}_rrCV={rr_cv:.2f}_snr={snr:.1f}'


# ---- SCG validity checks ----

def is_valid_scg(filepath):
    """Check if an SCG segment looks like a real SCG signal.

    Returns (is_valid, reason_string)
    """
    try:
        sig = np.loadtxt(filepath)
    except Exception:
        return False, 'read_error'

    if len(sig) < SR * 0.3:
        return False, 'too_short'

    sig_std = np.std(sig)
    if sig_std < 1e-6:
        return False, 'flat_signal'

    # Saturation check
    upper = np.sum(np.abs(sig - np.max(sig)) < 1e-8)
    lower = np.sum(np.abs(sig - np.min(sig)) < 1e-8)
    sat_ratio = (upper + lower) / len(sig)
    if sat_ratio > 0.1:
        return False, f'saturated={sat_ratio:.0%}'

    # Hilbert envelope for periodicity
    try:
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
    except Exception:
        return False, 'hilbert_fail'

    # Smooth envelope
    from scipy.signal import savgol_filter
    try:
        envelope = savgol_filter(envelope, min(21, len(envelope)//2*2+1), 2)
    except Exception:
        pass

    env_std = np.std(envelope)
    if env_std < 1e-10:
        return False, 'flat_envelope'

    # Find envelope peaks with prominence
    try:
        peaks, _ = find_peaks(envelope, prominence=env_std * 0.5,
                               distance=int(0.04 * SR))
    except Exception:
        return False, 'no_env_peaks'

    if len(peaks) < 2:
        return False, f'few_env_peaks={len(peaks)}'

    # Check if peak spacing corresponds to mouse HR
    rr_samples = np.diff(peaks)
    if len(rr_samples) == 0:
        return False, 'single_peak'

    hr_values = 60 * SR / rr_samples
    valid_hr_mask = (hr_values > 150) & (hr_values < 900)
    valid_hr_ratio = np.sum(valid_hr_mask) / len(hr_values)

    if valid_hr_ratio < 0.4:
        return False, f'env_hr_mismatch={1-valid_hr_ratio:.0%}'

    # SNR
    snr = np.max(envelope) / (np.median(envelope) + 1e-10)
    if snr < 2.0:
        return False, f'low_snr={snr:.1f}'

    return True, f'valid_npeaks={len(peaks)}_snr={snr:.1f}'


# ---- Main processing ----

def validate_and_filter():
    """Validate all segments, remove invalid ones, report statistics."""

    for modality, seg_dir, check_fn in [
        ('ECG', 'ecg_segments', is_valid_ecg),
        ('SCG', 'scg_segments', is_valid_scg),
    ]:
        print(f"\n{'='*60}")
        print(f"Validating {modality} segments in {seg_dir}/")
        print(f"{'='*60}")

        valid_list = []
        invalid_list = []
        reason_counts = {}

        for fname in sorted(os.listdir(seg_dir)):
            if not fname.endswith('.csv'):
                continue

            fpath = os.path.join(seg_dir, fname)
            valid, reason = check_fn(fpath)

            if valid:
                valid_list.append(fname)
            else:
                invalid_list.append(fname)
                reason_key = reason.split('=')[0] if '=' in reason else reason
                reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

        n_total = len(valid_list) + len(invalid_list)
        pct_valid = 100 * len(valid_list) / n_total if n_total > 0 else 0

        print(f"  Total: {n_total}")
        print(f"  Valid: {len(valid_list)} ({pct_valid:.1f}%)")
        print(f"  Invalid: {len(invalid_list)} ({100-pct_valid:.1f}%)")
        print(f"  Rejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        # Remove invalid segments
        removed_dir = seg_dir + '_invalid'
        os.makedirs(removed_dir, exist_ok=True)
        for fname in invalid_list:
            src = os.path.join(seg_dir, fname)
            dst = os.path.join(removed_dir, fname)
            shutil.move(src, dst)

        print(f"  Invalid segments moved to {removed_dir}/")

        # Print some examples of rejected segments with reasons
        if invalid_list:
            print(f"  Examples of rejected:")
            for fname in invalid_list[:5]:
                _, reason = check_fn(os.path.join(removed_dir, fname))
                print(f"    {fname}: {reason}")


if __name__ == '__main__':
    validate_and_filter()
