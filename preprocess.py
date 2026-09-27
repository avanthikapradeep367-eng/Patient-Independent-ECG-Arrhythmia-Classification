"""
Module 1 + 2 — Dataset loading & Preprocessing
ECG Patient-Independent Arrhythmia Classification project

Run this after: pip install wfdb numpy scipy scikit-learn

Output (share these 3 files with your teammates):
    X.npy            -> shape (num_beats, 250)  float32, normalized beat windows
    y.npy            -> shape (num_beats,)      int labels (0-5)
    patient_ids.npy  -> shape (num_beats,)      int, MIT-BIH record number per beat
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data"                 # where MIT-BIH records live / get downloaded
FS = 360                          # MIT-BIH sampling rate (Hz)
PRE_SAMPLES = 90                  # samples before R-peak
POST_SAMPLES = 160                # samples after R-peak
BEAT_LEN = PRE_SAMPLES + POST_SAMPLES   # = 250

# Standard AAMI/de Chazal patient-wise split (do not change these lists)
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,
       207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,
       221,222,228,231,232,233,234]
ALL_RECORDS = sorted(DS1 + DS2)

# MIT-BIH annotation symbol -> class index
# N: normal, L/R: bundle branch block, A: APB, V: PVC, P: paced
SYMBOL_TO_CLASS = {
    'N': 0, 'L': 1, 'R': 1,
    'A': 2,
    'V': 3,
    'P': 4,
    'e': 0, 'j': 0,   # variants of normal, folded in
}
CLASS_NAMES = ['Normal', 'BBB (L/R)', 'APB', 'PVC', 'Paced']


# ---------------------------------------------------------------------------
# Step 1: download dataset if not already present
# ---------------------------------------------------------------------------
def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    sample_file = os.path.join(DATA_DIR, "100.dat")
    if not os.path.exists(sample_file):
        print("Downloading MIT-BIH Arrhythmia Database ...")
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR)
    else:
        print("Dataset already present, skipping download.")


# ---------------------------------------------------------------------------
# Step 2: filtering
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Step 3: per-record beat extraction
# ---------------------------------------------------------------------------
def extract_beats_from_record(record_id, data_dir=DATA_DIR):
    """Returns (beats, labels) for one MIT-BIH record."""
    record_path = os.path.join(data_dir, str(record_id))
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    # MLII is channel 0 in almost all MIT-BIH records; fall back to channel 0
    ch = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
    raw_signal = record.p_signal[:, ch]

    filtered = bandpass_filter(raw_signal)

    beats, labels = [], []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in SYMBOL_TO_CLASS:
            continue  # skip non-beat annotations (rhythm change markers, etc.)

        start = sample - PRE_SAMPLES
        end = sample + POST_SAMPLES
        if start < 0 or end > len(filtered):
            continue  # skip beats too close to signal edges

        beat = filtered[start:end]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)  # per-beat z-score

        beats.append(beat)
        labels.append(SYMBOL_TO_CLASS[symbol])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Step 4: run over all records, save outputs
# ---------------------------------------------------------------------------
def build_dataset():
    ensure_dataset()

    all_X, all_y, all_pid = [], [], []

    for record_id in ALL_RECORDS:
        try:
            beats, labels = extract_beats_from_record(record_id)
        except Exception as e:
            print(f"  [WARN] record {record_id} failed: {e}")
            continue

        if len(beats) == 0:
            print(f"  [WARN] record {record_id} produced 0 beats")
            continue

        all_X.append(beats)
        all_y.append(labels)
        all_pid.append(np.full(len(beats), record_id, dtype=np.int64))
        print(f"  record {record_id}: {len(beats)} beats extracted")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    patient_ids = np.concatenate(all_pid, axis=0)

    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("patient_ids.npy", patient_ids)

    print("\nSaved X.npy, y.npy, patient_ids.npy")
    print(f"Total beats: {len(y)}")
    print("Class distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == idx))
        print(f"  {name:12s}: {count}")

    # sanity check the split lists don't overlap
    overlap = set(DS1) & set(DS2)
    assert not overlap, f"DS1/DS2 overlap detected: {overlap}"
    print("\nDS1/DS2 patient lists confirmed non-overlapping.")

    return X, y, patient_ids


if __name__ == "__main__":
    build_dataset()
