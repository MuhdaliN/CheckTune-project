# ml/utils.py
import librosa
import numpy as np

def extract_mfcc(wav_path, n_mfcc=40, sr=22050, max_len=174):
    """
    Return MFCC feature (n_mfcc x max_len) padded/truncated to max_len.
    max_len depends on chosen audio duration and hop_length; 174 is an example.
    """
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    # get mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    # shape => (n_mfcc, max_len)
    return mfcc.astype(np.float32)
