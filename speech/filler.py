import librosa
import numpy as np


def _energy_vad(y: np.ndarray, sr: int, hop_length: int = 256, threshold_db: float = -45) -> list:
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced = db > threshold_db

    segments = []
    in_seg = False
    start = 0

    for i, v in enumerate(voiced):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append({"start": start * hop_length / sr,
                              "end": i * hop_length / sr})
            in_seg = False

    if in_seg:
        segments.append({"start": start * hop_length / sr,
                          "end": len(voiced) * hop_length / sr})

    merged = []
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] < 0.1:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    return merged


def _is_vowel_like(y_seg: np.ndarray, sr: int) -> bool:
    if len(y_seg) < 100:
        return False
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_seg)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr)))
    return zcr < 0.06 and 200 < centroid < 3000


def _classify_filler(y_seg: np.ndarray, sr: int) -> str:
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr)))
    if centroid < 800:
        return "음"
    elif centroid < 1400:
        return "어"
    elif centroid < 2000:
        return "오"
    else:
        return "아"


def _frequent_fillers(fillers: list) -> list:
    counts: dict = {}
    for f in fillers:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    return sorted(counts, key=lambda k: counts[k], reverse=True)


def detect_fillers(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    segments = _energy_vad(y, sr)

    fillers = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]

        if not (0.75 <= duration <= 2.5):
            continue

        prev_long = i > 0 and (segments[i - 1]["end"] - segments[i - 1]["start"]) > 0.75
        next_long = i < len(segments) - 1 and (segments[i + 1]["end"] - segments[i + 1]["start"]) > 0.75

        if not (prev_long and next_long):
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        y_seg = y[start_sample:end_sample]

        if _is_vowel_like(y_seg, sr):
            fillers.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "duration": round(duration, 2),
                "type": _classify_filler(y_seg, sr),
            })

    return {
        "filler_count": len(fillers),
        "frequent_fillers": _frequent_fillers(fillers),
        "fillers": fillers,
    }