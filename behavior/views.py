import os,cv2
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from django.db.models import Avg, Sum
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiExample

from interview.models import Interview, InterviewQuestion
from .models import BehaviorDetail
from .analysis_logic import analyze_behavior_video
# from speech.analyzer import analyze_question
# from speech.aggregator import create_speech_interview_report

from .serializers import (
    VideoUploadSerializer,
    VideoUploadResponseSerializer,
    CumulativeTrendsResponseSerializer
)

# ===========================================================================
# [2번 API] 질문별 답변 영상 업로드 및 분석 요청 (POST)
# Endpoint: /interviews/<int:interview_id>/questions/
# ===========================================================================
@extend_schema(
    summary="[2] 질문별 답변 영상 업로드 및 분석 요청",
    description="각 질문 응답 영상을 전송받아 실시간 프레임 및 통계를 저장합니다.",
    request=VideoUploadSerializer,                 
    responses={202: VideoUploadResponseSerializer},
    examples=[
        OpenApiExample(
            '2. 질문 업로드 Response 예시',
            value={
                "interview_id": 1,
                "question_order": 1,
                "status": "ANALYZING",
                "message": "Video uploaded successfully. Analysis started."
            },
            response_only=True,
            status_codes=['202'] # 🌟 Swagger 0 뭉개짐 방지 매핑
        )
    ],
    tags=['1. Interviews']
)
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def process_video_analysis(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id)
    if not interview.calibration_config:
        return Response({"error": "캘리브레이션(초기 세팅)이 완료되지 않은 세션입니다."}, status=400)

    video_file = request.FILES.get('video_file')
    question_order = request.data.get('question_order')
    question_text = request.data.get('question_text')

    if not video_file or not question_order or not question_text:
        return Response({"error": "필수 데이터 누락"}, status=400)

    order = int(question_order)

    ext = os.path.splitext(video_file.name)[1]
    saved_file_name = default_storage.save(f"videos/intv_{interview_id}_{order}{ext}", ContentFile(video_file.read()))
    full_video_path = os.path.join(settings.MEDIA_ROOT, saved_file_name)

    question_obj, _ = InterviewQuestion.objects.update_or_create(
        interview=interview, order=order,
        defaults={'question_text': question_text, 'video_path': saved_file_name, 'status': 'ANALYZING'}
    )

    try:
        frame_details, summary = analyze_behavior_video(full_video_path, config=interview.calibration_config)

        details_to_create = [
            BehaviorDetail(
                question=question_obj,
                timestamp=d['timestamp'],
                gaze_direction=d.get('gaze_direction', 'center'),
                is_swaying=d.get('is_swaying', False),
                is_blink=d.get('is_blink', False),
                is_nodding=d.get('is_nodding', False),
                is_smiling=d.get('is_smiling', False),
                shoulder_stable_frame=d.get('shoulder_stable', True)
            ) for d in frame_details
        ]
        BehaviorDetail.objects.bulk_create(details_to_create)

        duration_sec = summary.get('duration_sec', 0.0)
        
        question_obj.gaze_front_ratio = summary.get('focus_rate', 0.0)
        question_obj.gaze_deviation_ratio = summary.get('deviated_gaze_rate', 0.0)
        question_obj.body_sway_count = summary.get('lr_sway_count', 0)
        question_obj.shoulder_stability_ratio = summary.get('shoulder_stability', 100.0)
        
        raw_blink_pm = summary.get('blinks_per_min')
        question_obj.blink_count = int(raw_blink_pm * (duration_sec / 60)) if raw_blink_pm is not None else 0
        
        question_obj.nod_count = summary.get('nod_count', 0)
        question_obj.smile_ratio = summary.get('total_smile_rate', 0.0)
        question_obj.video_duration = duration_sec
        question_obj.status = 'COMPLETED'
        question_obj.save()

        try:
            analyze_question(question_obj)
        except Exception as e:
            print(f"[Speech] 음성 분석 실패 (question {order}): {e}")

        all_completed = interview.questions.filter(status='COMPLETED').count() >= interview.question_count
        if all_completed:
            interview.status = 'COMPLETED'
            interview.save()
            try:
                create_speech_interview_report(interview)
            except Exception as e:
                print(f"[Speech] 인터뷰 집계 리포트 생성 실패: {e}")

        # 응답 구조 명세화하여 리턴
        out_data = {
            "interview_id": interview.id,
            "question_order": order,
            "status": "ANALYZING",
            "message": "Video uploaded successfully. Analysis started."
        }
        return Response(out_data, status=status.HTTP_202_ACCEPTED)

    except Exception as e:
        question_obj.status = 'FAILED'
        question_obj.save()
        return Response({"error": f"분석 실패: {str(e)}"}, status=500)


# ===========================================================================
# [5번 API] 회차별 변화 트렌드 데이터 조회 (GET)
# Endpoint: /interviews/trends/
# ===========================================================================
@extend_schema(
    summary="[5] 회차별 변화 트렌드 데이터 조회",
    description="도표/그래프를 그리기 위해 과거 역대 면접들의 핵심 분석 수치들을 시간 순서대로 반환합니다.",
    responses={200: CumulativeTrendsResponseSerializer},
    examples=[
        OpenApiExample(
            '5. 시계열 변화 트렌드 Response 예시',
            value={
                "total_interview_count": 2,
                "trends": [
                    {
                        "interview_id": 1,
                        "date": "2026-05-20",
                        "interview_type": "JOB",
                        "gaze_front_ratio": 72.1,
                        "body_sway_per_min": 5.4,
                        "blink_per_min": 22.1,
                        "smile_ratio": 5.0
                    },
                    {
                        "interview_id": 2,
                        "date": "2026-05-25",
                        "interview_type": "RESUME",
                        "gaze_front_ratio": 85.5,
                        "body_sway_per_min": 2.3,
                        "blink_per_min": 15.2,
                        "smile_ratio": 12.5
                    }
                ]
            },
            response_only=True
        )
    ],
    tags=['1. Interviews']
)
@api_view(['GET'])
def get_interview_trends(request):
    completed_interviews = Interview.objects.filter(status='COMPLETED').order_by('created_at')
    
    trends_list = []
    
    for intv in completed_interviews:
        questions = intv.questions.filter(status='COMPLETED')
        if not questions.exists():
            continue
            
        total_duration = questions.aggregate(Sum('video_duration'))['video_duration__sum'] or 0.0
        total_duration_min = total_duration / 60 if total_duration > 0 else 1.0
        
        avg_gaze_front = questions.aggregate(Avg('gaze_front_ratio'))['gaze_front_ratio__avg'] or 0.0
        avg_smile = questions.aggregate(Avg('smile_ratio'))['smile_ratio__avg'] or 0.0
        
        total_sways = questions.aggregate(Sum('body_sway_count'))['body_sway_count__sum'] or 0
        total_blinks = questions.aggregate(Sum('blink_count'))['blink_count__sum'] or 0
        
        body_sway_per_min = total_sways / total_duration_min
        blink_per_min = total_blinks / total_duration_min
        
        trends_list.append({
            "interview_id": intv.id,
            "date": intv.created_at.strftime("%Y-%m-%d"),
            "interview_type": intv.interview_type,
            "gaze_front_ratio": round(avg_gaze_front, 1),
            "body_sway_per_min": round(body_sway_per_min, 1),
            "blink_per_min": round(blink_per_min, 1),
            "smile_ratio": round(avg_smile, 1)
        })
        
    return Response({
        "total_interview_count": len(trends_list),
        "trends": trends_list
    }, status=status.HTTP_200_OK)