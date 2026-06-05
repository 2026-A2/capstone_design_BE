import os
import ffmpeg
from django.db import IntegrityError
from speech.models import SpeechAnalysis, SpeechReport, SpeechSilence, SpeechFiller
from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _extract_audio(video_path: str) -> str:
    wav_path = video_path + "_audio.wav"
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


def analyze_question(question) -> dict:
    """InterviewQuestion 하나에 대해 음성 분석 수행 후 DB 저장 및 결과 dict 반환."""
    try:
        _ = question.speech_analysis
        raise ValueError(f"question {question.id}는 이미 음성 분석이 완료되었습니다.")
    except SpeechAnalysis.DoesNotExist:
        pass

    file_path = question.video_path.path
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    extracted_path = None
    try:
        if ext in VIDEO_EXTENSIONS:
            extracted_path = _extract_audio(file_path)
            analyze_path = extracted_path
        else:
            analyze_path = file_path

        result = analyze(analyze_path)
        result.pop("segments", None)

        volume = analyze_volume(analyze_path)
        filler = detect_fillers(analyze_path)

        result["volume"] = volume
        result["filler"] = filler

        try:
            speech_analysis = SpeechAnalysis.objects.create(question=question)
        except IntegrityError:
            raise ValueError(f"question {question.id}는 이미 음성 분석이 완료되었습니다.")

        SpeechReport.objects.create(
            analysis=speech_analysis,
            transcript=result["transcript"],
            syllable_count=result["syllable_count"],
            duration_sec=result["duration_sec"],
            spm=result["spm"],
            pace=result["pace"],
            avg_db=volume["avg_db"],
            max_db=volume["max_db"],
            min_db=volume["min_db"],
            std_db=volume["std_db"],
            volume_level=volume["volume_level"],
            filler_count=filler["filler_count"],
            frequent_fillers=filler["frequent_fillers"],
            volume_timeline=volume["volume_timeline"],
        )

        SpeechSilence.objects.bulk_create([
            SpeechSilence(analysis=speech_analysis, start=s["start"], end=s["end"], duration=s["duration"])
            for s in result["silences"]
        ])

        SpeechFiller.objects.bulk_create([
            SpeechFiller(analysis=speech_analysis, start=f["start"], end=f["end"], duration=f["duration"], filler_type=f["type"])
            for f in filler["fillers"]
        ])

        return {"order": question.order, "file_name": file_name, **result}

    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)
