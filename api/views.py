import os
import tempfile

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from analyzer import analyze
from analyzer2 import analyze_volume
from analyzer3 import detect_fillers

SUPPORTED_EXTENSIONS = {".wav", ".mp4", ".m4a"}


@api_view(["POST"])
def analyze_audio(request):
    if "audio" not in request.FILES:
        return Response(
            {"error": "음성 파일을 'audio' 키로 전송해 주세요."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    audio_file = request.FILES["audio"]
    ext = os.path.splitext(audio_file.name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return Response(
            {"error": f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        result = analyze(tmp_path)
        segments = result.pop("segments", None)

        result["volume"] = analyze_volume(tmp_path, segments=segments)
        result["filler"] = detect_fillers(tmp_path)

        return Response(result)
    except Exception as e:
        return Response(
            {"error": f"분석 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        os.unlink(tmp_path)
