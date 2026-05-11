import os
import tempfile

import ffmpeg
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers
from speech.models import SpeechAnalysis, SpeechReport, SpeechSilence, SpeechFiller

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio(video_path: str) -> str:
    wav_path = video_path + "_audio.wav"
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


@api_view(["POST"])
def analyze_audio(request):
    if "audio" not in request.FILES:
        return Response(
            {"error": "음성/영상 파일을 'audio' 키로 전송해 주세요."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    audio_file = request.FILES["audio"]
    ext = os.path.splitext(audio_file.name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return Response(
            {"error": f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    extracted_path = None
    try:
        if ext in VIDEO_EXTENSIONS:
            extracted_path = extract_audio(tmp_path)
            analyze_path = extracted_path
        else:
            analyze_path = tmp_path

        result = analyze(analyze_path)
        segments = result.pop("segments", None)

        volume = analyze_volume(analyze_path, segments=segments)
        filler = detect_fillers(analyze_path)

        result["volume"] = volume
        result["filler"] = filler

        speech_analysis = SpeechAnalysis.objects.create(file_name=audio_file.name)

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
            SpeechSilence(
                analysis=speech_analysis,
                start=s["start"],
                end=s["end"],
                duration=s["duration"],
            )
            for s in result["silences"]
        ])

        SpeechFiller.objects.bulk_create([
            SpeechFiller(
                analysis=speech_analysis,
                start=f["start"],
                end=f["end"],
                duration=f["duration"],
                filler_type=f["type"],
            )
            for f in filler["fillers"]
        ])

        return Response(result)
    except Exception as e:
        return Response(
            {"error": f"분석 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        os.unlink(tmp_path)
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)
