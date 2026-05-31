from speech.transcriber import calculate_speech_rate
from speech.volume import _classify as classify_volume_level


def create_speech_interview_report(interview):
    """모든 질문의 SpeechReport/SpeechFiller/SpeechSilence를 DB에서 읽어 SpeechInterviewReport를 생성합니다."""
    from speech.models import SpeechAnalysis, SpeechReport, SpeechFiller, SpeechSilence, SpeechInterviewReport

    analyses = SpeechAnalysis.objects.filter(
        question__interview=interview
    ).prefetch_related('report', 'fillers', 'silences')

    reports = [a.report for a in analyses if hasattr(a, 'report')]
    if not reports:
        return None

    total_syllables = sum(r.syllable_count for r in reports)
    total_duration = sum(r.duration_sec for r in reports)
    rate_info = calculate_speech_rate("가" * total_syllables, total_duration)

    avg_db = round(sum(r.avg_db for r in reports) / len(reports), 2)
    max_db = round(max(r.max_db for r in reports), 2)
    min_db = round(min(r.min_db for r in reports), 2)
    avg_std_db = round(sum(r.std_db for r in reports) / len(reports), 2)
    volume_level = classify_volume_level(avg_db)

    total_filler_count = sum(r.filler_count for r in reports)
    filler_counts: dict = {}
    for a in analyses:
        for f in a.fillers.all():
            filler_counts[f.filler_type] = filler_counts.get(f.filler_type, 0) + 1
    frequent_fillers = sorted(filler_counts, key=lambda k: filler_counts[k], reverse=True)

    all_silences = [s for a in analyses for s in a.silences.all()]
    total_silence_count = len(all_silences)
    avg_silence_duration = (
        round(sum(s.duration for s in all_silences) / total_silence_count, 2)
        if total_silence_count > 0 else 0.0
    )

    return SpeechInterviewReport.objects.create(
        interview=interview,
        avg_spm=rate_info["spm"],
        pace=rate_info["pace"],
        avg_db=avg_db,
        max_db=max_db,
        min_db=min_db,
        avg_std_db=avg_std_db,
        volume_level=volume_level,
        total_filler_count=total_filler_count,
        frequent_fillers=frequent_fillers,
        total_silence_count=total_silence_count,
        avg_silence_duration=avg_silence_duration,
    )


def aggregate_results(results: list) -> dict:
    """
    n개 영상 분석 결과를 받아 항목별 집계 방식으로 최종 요약을 반환한다.
    results: views.py에서 각 영상 분석 후 모은 dict 리스트
    """
    total_syllables = sum(r["syllable_count"] for r in results)
    total_duration = sum(r["duration_sec"] for r in results)

    rate_info = calculate_speech_rate(
        "가" * total_syllables,  # 음절 수만 맞추기 위한 더미 텍스트
        total_duration,
    )
    avg_spm = rate_info["spm"]
    pace = rate_info["pace"]

    avg_db = round(sum(r["volume"]["avg_db"] for r in results) / len(results), 2)
    max_db = round(max(r["volume"]["max_db"] for r in results), 2)
    min_db = round(min(r["volume"]["min_db"] for r in results), 2)
    avg_std_db = round(sum(r["volume"]["std_db"] for r in results) / len(results), 2)
    volume_level = classify_volume_level(avg_db)

    total_filler_count = sum(r["filler"]["filler_count"] for r in results)
    frequent_fillers = _merge_frequent_fillers(results)

    all_silences = [s for r in results for s in r["silences"]]
    total_silence_count = len(all_silences)
    avg_silence_duration = (
        round(sum(s["duration"] for s in all_silences) / total_silence_count, 2)
        if total_silence_count > 0 else 0.0
    )

    return {
        "avg_spm": avg_spm,
        "pace": pace,
        "avg_db": avg_db,
        "max_db": max_db,
        "min_db": min_db,
        "avg_std_db": avg_std_db,
        "volume_level": volume_level,
        "total_filler_count": total_filler_count,
        "frequent_fillers": frequent_fillers,
        "total_silence_count": total_silence_count,
        "avg_silence_duration": avg_silence_duration,
    }


def _merge_frequent_fillers(results: list) -> list:
    counts: dict = {}
    for r in results:
        for filler in r["filler"]["fillers"]:
            counts[filler["type"]] = counts.get(filler["type"], 0) + 1
    return sorted(counts, key=lambda k: counts[k], reverse=True)
