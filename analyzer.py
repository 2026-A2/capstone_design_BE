import whisper


def count_syllables(text: str) -> int:
    return sum(1 for ch in text if '가' <= ch <= '힣')


def transcribe(audio_path: str, model) -> dict:
    return model.transcribe(audio_path, word_timestamps=True)


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
    print("Loading Whisper base model...")
    model = whisper.load_model("base")

    result = transcribe(audio_path, model)
    transcript = result["text"].strip()
    duration_sec = result["segments"][-1]["end"] if result["segments"] else 0

    rate_info = calculate_speech_rate(transcript, duration_sec)
    silences = detect_silences(result["segments"])

    return {
        "transcript": transcript,
        "silences": silences,
        "segments": result["segments"],
        **rate_info,
    }