import re
from speech.transcriber import transcribe_for_fillers

FILLER_WORDS = {"음", "어", "아", "그"}

_PUNCT_RE = re.compile(r'[^가-힣\w]')


def _clean_word(word: str) -> str:
    return _PUNCT_RE.sub('', word.strip())


def detect_fillers(audio_path: str) -> dict:
    segments = transcribe_for_fillers(audio_path)

    fillers = []
    for seg in segments:
        for word_info in seg.get("words", []):
            cleaned = _clean_word(word_info["word"])
            if cleaned not in FILLER_WORDS:
                continue
            duration = word_info["end"] - word_info["start"]
            if duration < 0.15:
                continue
            if word_info.get("probability", 1.0) < 0.6:
                continue
            fillers.append({
                "start": round(word_info["start"], 2),
                "end": round(word_info["end"], 2),
                "duration": round(duration, 2),
                "type": cleaned,
            })

    filler_counts: dict = {}
    for f in fillers:
        filler_counts[f["type"]] = filler_counts.get(f["type"], 0) + 1
    frequent_fillers = sorted(filler_counts, key=lambda k: filler_counts[k], reverse=True)

    return {
        "filler_count": len(fillers),
        "frequent_fillers": frequent_fillers,
        "fillers": fillers,
    }
