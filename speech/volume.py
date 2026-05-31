import librosa
import numpy as np


def analyze_volume(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)

    timeline = [
        {"time_sec": round(float(t), 2), "db": round(float(d), 2)}
        for t, d in zip(times, db)
    ]

    return {
        "avg_db": round(float(np.mean(db)), 2),
        "max_db": round(float(np.max(db)), 2),
        "min_db": round(float(np.min(db)), 2),
        "std_db": round(float(np.std(db)), 2),
        "volume_level": _classify(float(np.mean(db))),
        "volume_timeline": timeline,
    }


def _classify(avg_db: float) -> str:
    if avg_db >= -10:
        return "매우 큼"
    elif avg_db >= -20:
        return "큼"
    elif avg_db >= -35:
        return "보통"
    elif avg_db >= -50:
        return "작음"
    else:
        return "매우 작음"