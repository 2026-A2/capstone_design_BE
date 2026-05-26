import os
import traceback
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from django.db.models import Avg, Sum
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser # 🌟 파일 파서 추가
from drf_spectacular.utils import extend_schema, OpenApiExample

from .models import Interview
from .serializers import (
    InterviewCreateSerializer, 
    InterviewResponseSerializer,
    InterviewListElementSerializer,
    FinalReportSerializer
)
# 🌟 behavior 앱에 선언된 캘리브레이션 연산 로직 호출
from behavior.analysis_logic import run_calibration 

# ===========================================================================
# [통합 1번] 면접 세션 생성 + 초기 캘리브레이션 동시 처리 (POST)
# [4번] 전체 면접 리포트 목록 조회 (GET)
# Endpoint: /interviews/
# ===========================================================================
@extend_schema(
    methods=['POST'],
    summary="[1] 면접 세션 생성 & 캘리브레이션 파일 업로드 통합",
    description="면접 설정 데이터와 초기 환경 측정용 캘리브레이션 영상을 동시에 전송받아 세션을 생성하고 분석 프로필을 빌드합니다.",
    request=InterviewCreateSerializer,
    responses={201: InterviewResponseSerializer},
    examples=[
        OpenApiExample(
            '1. 통합 생성 및 업로드 Response 예시',
            value={
                "interview_id": 1,
                "interview_type": "RESUME",
                "question_count": 5,
                "status": "CALIBRATED" # 생성되자마자 캘리브레이션까지 완료됨을 명시
            },
            response_only=True
        )
    ],
    tags=['1. Interviews']
)
@extend_schema(
    methods=['GET'],
    summary="[4] 전체 면접 리포트 목록 조회",
    description="과거에 진행했던 모든 면접 기록 리스트를 최신 날짜 역순으로 가져옵니다.",
    responses={200: InterviewListElementSerializer(many=True)},
    examples=[
        OpenApiExample(
            '4. 역대 목록 Response 예시',
            value=[
                {"interview_id": 2, "interview_type": "RESUME", "created_at": "2026-05-25", "question_count": 5, "status": "COMPLETED"},
                {"interview_id": 1, "interview_type": "JOB", "created_at": "2026-05-20", "question_count": 3, "status": "COMPLETED"}
            ],
            response_only=True
        )
    ],
    tags=['1. Interviews']
)
# 파일 업로드를 받아들이기 위해 멀티파트 파서 지정
@api_view(['POST', 'GET'])
@parser_classes([MultiPartParser, FormParser]) 
def interview_base_handler(request):
    if request.method == 'POST':
        # 파일이 믹스된 데이터를 유효성 검사
        serializer = InterviewCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        validated_data = serializer.validated_data
        video_file = request.FILES.get('video_file')
        
        if not video_file:
            return Response({"error": "캘리브레이션 영상 파일이 누락되었습니다."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. 먼저 DB에 면접 기본 데이터 적재 (중간 저장)
        interview = Interview.objects.create(
            interview_type=validated_data['interview_type'],
            resume_text=validated_data.get('resume_text', ''),
            job_category=validated_data.get('job_category', ''),
            question_count=validated_data['question_count'],
            status='CREATED'
        )

        # 2. 업로드된 파일 디스크 저장 처리
        ext = os.path.splitext(video_file.name)[1]
        saved_file_name = default_storage.save(f"videos/calib_{interview.id}{ext}", ContentFile(video_file.read()))
        full_video_path = os.path.join(settings.MEDIA_ROOT, saved_file_name)

        # 3. AI 엔진 연산 작동 시키기 (기존 1-2번 코드를 흡수)
        try:
            calibration_profile = run_calibration(full_video_path)
            interview.calibration_config = calibration_profile
            interview.status = 'CALIBRATED' # 성공적으로 연산 완료 시 세션 레벨 업!
            interview.save()

            response_serializer = InterviewResponseSerializer(interview)
            
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            print("===== ERROR =====")
            traceback.print_exc()

            interview.delete()

            return Response({
                "error": str(e)
            }, status=500)
        
    elif request.method == 'GET':
        interviews = Interview.objects.all().order_by('-created_at')
        response_data = [
            {
                "interview_id": intv.id,
                "interview_type": intv.interview_type,
                "created_at": intv.created_at.strftime("%Y-%m-%d"),
                "question_count": intv.question_count,
                "status": intv.status
            } for intv in interviews
        ]
        return Response(response_data, status=status.HTTP_200_OK)
        # return Response([], status=status.HTTP_200_OK)

# ===========================================================================
# 3. 방금 마친 면접 결과 상세 리포트 조회 (GET)
# Endpoint: /interviews/<int:interview_id>/report/
# ===========================================================================
@extend_schema(
    summary="[3] 방금 마친 면접 결과 상세 리포트 조회",
    description="면접과 모든 분석이 완료된 직후, 결과 화면에서 특정 면접 세션의 행동 상세 분석 값을 가져옵니다.",
    responses={200: FinalReportSerializer},
    examples=[
        OpenApiExample(
            '3. 상세 리포트 Response 예시',
            value={
                "interview_id": 1,
                "interview_type": "RESUME",
                "status": "COMPLETED",
                "question_count": 5,
                "total_video_duration": 540,
                "created_at": "2026-05-25T14:30:00",
                "analysis_result": {
                    "behavior": {
                        "gaze_front_ratio": 85.5,
                        "gaze_deviation_ratio": 14.5,
                        "body_sway_per_min": 2.3,
                        "shoulder_stability": 92.5,
                        "blink_per_min": 15.2,
                        "nod_per_min": 4.1,
                        "smile_ratio": 12.5
                    },
                    "speech": {
                        "avg_spm": 312.4,
                        "pace": "보통",
                        "avg_db": -18.3,
                        "volume_level": "보통",
                        "total_filler_count": 7,
                        "frequent_fillers": ["어", "음", "그"],
                        "total_silence_count": 3,
                        "avg_silence_duration": 4.2
                    }
                }
            },
            response_only=True
        )
    ],
    tags=['1. Interviews']
)
@api_view(['GET']) 
def get_final_report(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id)
    completed_questions = interview.questions.filter(status='COMPLETED')
    
    if not completed_questions.exists():
        return Response({
            "status": "not_found",
            "message": "분석 완료된 질문 데이터가 없습니다. 영상 업로드 상태를 확인하세요."
        }, status=status.HTTP_400_BAD_REQUEST)

    total_duration = completed_questions.aggregate(Sum('video_duration'))['video_duration__sum'] or 0.0
    total_duration_min = total_duration / 60 if total_duration > 0 else 1.0
    
    avg_gaze_front = completed_questions.aggregate(Avg('gaze_front_ratio'))['gaze_front_ratio__avg'] or 0.0
    avg_gaze_dev = completed_questions.aggregate(Avg('gaze_deviation_ratio'))['gaze_deviation_ratio__avg'] or 0.0
    avg_shoulder = completed_questions.aggregate(Avg('shoulder_stability_ratio'))['shoulder_stability_ratio__avg'] or 0.0
    avg_smile = completed_questions.aggregate(Avg('smile_ratio'))['smile_ratio__avg'] or 0.0
    
    total_sways = completed_questions.aggregate(Sum('body_sway_count'))['body_sway_count__sum'] or 0
    total_blinks = completed_questions.aggregate(Sum('blink_count'))['blink_count__sum'] or 0
    total_nods = completed_questions.aggregate(Sum('nod_count'))['nod_count__sum'] or 0
    
    body_sway_per_min = total_sways / total_duration_min
    blink_per_min = total_blinks / total_duration_min
    nod_per_min = total_nods / total_duration_min

    speech_data = {}
    if hasattr(interview, 'speech_report'):
        sr = interview.speech_report
        speech_data = {
            "avg_spm": sr.avg_spm,
            "pace": sr.pace,
            "avg_db": sr.avg_db,
            "volume_level": sr.volume_level,
            "total_filler_count": sr.total_filler_count,
            "frequent_fillers": sr.frequent_fillers,
            "total_silence_count": sr.total_silence_count,
            "avg_silence_duration": sr.avg_silence_duration,
        }

    return Response({
        "interview_id": interview.id,
        "interview_type": interview.interview_type,
        "status": interview.status,
        "question_count": interview.question_count,
        "total_video_duration": int(total_duration),
        "created_at": interview.created_at.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_result": {
            "behavior": {
                "gaze_front_ratio": round(avg_gaze_front, 1),
                "gaze_deviation_ratio": round(avg_gaze_dev, 1),
                "body_sway_per_min": round(body_sway_per_min, 1),
                "shoulder_stability": round(avg_shoulder, 1),
                "blink_per_min": round(blink_per_min, 1),
                "nod_per_min": round(nod_per_min, 1),
                "smile_ratio": round(avg_smile, 1)
            },
            "speech": speech_data,
        }
    }, status=status.HTTP_200_OK)