from faster_whisper import WhisperModel

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        print("Loading faster-whisper large-v3-turbo model...")
        _model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    return _model


def count_syllables(text: str) -> int:
    return sum(1 for ch in text if '가' <= ch <= '힣')


def _segments_to_dict(segments) -> list:
    """faster-whisper Segment 객체를 dict 형식으로 변환 (volume.py, filler.py 호환)"""
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (seg.words or [])
            ],
        })
    return result


def detect_silences(segments: list, threshold: float = 3.0) -> list:
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append(w)

    silences = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= threshold:
            silences.append({
                "start": round(words[i - 1]["end"], 2),
                "end": round(words[i]["start"], 2),
                "duration": round(gap, 2),
            })
    return silences


def calculate_speech_rate(transcript: str, duration_sec: float) -> dict:
    syllable_count = count_syllables(transcript)
    duration_min = duration_sec / 60.0
    spm = syllable_count / duration_min if duration_min > 0 else 0

    if spm < 250:
        pace = "느림"
    elif spm <= 350:
        pace = "보통"
    elif spm <= 450:
        pace = "빠름"
    else:
        pace = "매우 빠름"

    return {
        "syllable_count": syllable_count,
        "duration_sec": duration_sec,
        "spm": round(spm, 1),
        "pace": pace,
    }


def analyze(audio_path: str, **kwargs) -> dict:
    model = _get_model()

    segments_gen, info = model.transcribe(
        audio_path,
        language="ko",
        word_timestamps=True,
    )
    segments = _segments_to_dict(segments_gen)

    transcript = " ".join(seg["text"].strip() for seg in segments)
    duration_sec = info.duration

    rate_info = calculate_speech_rate(transcript, duration_sec)
    silences = detect_silences(segments)

    return {
        "transcript": transcript,
        "silences": silences,
        "segments": segments,
        **rate_info,
    }
