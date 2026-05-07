import librosa
import numpy as np


def analyze_volume(audio_path: str, segments: list = None) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)

    timeline = [
        {"time_sec": round(float(t), 2), "db": round(float(d), 2)}
        for t, d in zip(times, db)
    ]

    trailing_off = detect_trailing_off(y, sr, segments) if segments else []

    return {
        "avg_db": round(float(np.mean(db)), 2),
        "max_db": round(float(np.max(db)), 2),
        "min_db": round(float(np.min(db)), 2),
        "std_db": round(float(np.std(db)), 2),
        "volume_level": _classify(float(np.mean(db))),
        "trailing_off": trailing_off,
        "volume_timeline": timeline,
    }


def detect_trailing_off(y: np.ndarray, sr: int, segments: list,
                         end_duration: float = 0.5, threshold: float = 0.2) -> list:
    results = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        seg_audio = y[start_sample:end_sample]

        if len(seg_audio) == 0:
            continue

        seg_rms = float(np.sqrt(np.mean(seg_audio ** 2)))

        end_samples = int(end_duration * sr)
        end_audio = seg_audio[-end_samples:]

        if len(end_audio) == 0 or seg_rms == 0:
            continue

        end_rms = float(np.sqrt(np.mean(end_audio ** 2)))
        drop_ratio = (seg_rms - end_rms) / seg_rms

        if end_rms < threshold * seg_rms:
            results.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg.get("text", "").strip(),
                "감소율": round(drop_ratio * 100, 1),
            })

    return results


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