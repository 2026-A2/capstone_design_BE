import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiExample # OpenApiExample 누락 해결

from interview.models import Interview, InterviewQuestion
from .models import BehaviorReport # 모델 임포트 누락 해결
from .serializers import (
    VideoUploadSerializer, 
    BehaviorReportSerializer, 
    CumulativeReportSerializer
) 
from .analysis_logic import analyze_behavior_video
from .models import BehaviorDetail

# [2.1] 답변 영상 및 질문 등록
@extend_schema(
    summary="[2.1] 답변 영상 및 질문 등록",
    description="영상 파일은 로컬 'media/videos/' 폴더에 저장되고, 질문 텍스트는 DB에 저장됩니다.",
    request={
        'multipart/form-data': VideoUploadSerializer, # Swagger 파일 업로드 버튼 활성화
    },
    tags=['2. Analysis Flow']
)
# behavior/views.py

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def process_video_analysis(request, interview_id, order):
    interview = get_object_or_404(Interview, id=interview_id)
    video_file = request.FILES.get('video_file')
    question_text = request.data.get('question_text')

    if not video_file or not question_text:
        return Response({"error": "데이터 누락"}, status=400)

    # 1. 파일 저장
    ext = os.path.splitext(video_file.name)[1]
    new_filename = f"intv_{interview_id}_{order}{ext}"
    save_path = os.path.join('videos', new_filename)

    video_file.seek(0)
    if default_storage.exists(save_path):
        default_storage.delete(save_path)
    
    # 파일을 물리적으로 저장
    saved_file_name = default_storage.save(save_path, ContentFile(video_file.read()))
    # 분석을 위해 실제 물리 서버 경로(C:/... 또는 /Users/...)를 가져옴
    full_video_path = os.path.join(settings.MEDIA_ROOT, saved_file_name)

    # 2. 질문 데이터 DB 저장
    question_obj, created = InterviewQuestion.objects.update_or_create(
        interview=interview,
        order=order,
        defaults={'question_text': question_text}
    )

    # --- [핵심] 분석 엔진 가동 ---
    # 이 줄이 빠지면 frame_details가 정의되지 않아 에러가 납니다.
    frame_details, summary = analyze_behavior_video(full_video_path)

    # 3. 분석 결과(상세 데이터)를 DB에 대량 저장
    details_to_create = [
        BehaviorDetail(
            question=question_obj,
            timestamp=d['timestamp'],
            gaze_x=d.get('gaze_x', 0),
            gaze_y=d.get('gaze_y', 0),
            head_yaw=d.get('head_yaw', 0),
            head_pitch=d.get('head_pitch', 0),
            head_roll=d.get('head_roll', 0),
            shoulder_tilt=d.get('shoulder_tilt', 0),
            shoulder_width=d.get('shoulder_width', 0),
            center_x=d.get('center_x', 0),
            is_smiling=d.get('is_smiling', False),
            is_blink=d.get('is_blink', False),
            is_swaying=d.get('is_swaying', False),
            gaze_direction=d.get('gaze_direction', 'center')
        ) for d in frame_details
    ]
    BehaviorDetail.objects.bulk_create(details_to_create)

    # 4. 자동 상태 변경
    actual_question_count = interview.questions.count()
    is_completed = False
    if actual_question_count >= interview.question_count:
        interview.status = 'completed'
        interview.save()
        is_completed = True

    return Response({
        "message": f"{order}번 질문 분석 및 저장 완료",
        "current_order": order,
        "is_completed": is_completed
    }, status=status.HTTP_202_ACCEPTED)



# [3.1] 개별 리포트 상세 조회
@extend_schema(
    summary="[3.1] 개별 리포트 상세 조회",
    responses={200: BehaviorReportSerializer},
    examples=[
        OpenApiExample(
            '상세 리포트 응답 예시',
            value={
                "status": "success",
                "interview_id": 1,
                "questions": [
                    {"order": 1, "text": "지원동기를 말씀해주세요."},
                    {"order": 2, "text": "본인의 장점은 무엇인가요?"}
                ],
                "report": {
                    "focus_rate": 85.5,
                    "left_gaze_rate": 7.2,
                    "right_gaze_rate": 7.3,
                    "blinks_per_min": 12.0,
                    "nod_count": 5,
                    "shoulder_stability": 92.0,
                    "lr_sway_count": 2,
                    "fb_sway_count": 1,
                    "total_smile_rate": 45.0,
                    "start_smile_status": True,
                    "end_smile_status": True,
                    "overall_score": 88
                }
            },
            response_only=True
        )
    ],
    tags=['3. Report Flow']
)
@api_view(['GET'])
def get_individual_report(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id)
    report = get_object_or_404(BehaviorReport, interview=interview)
    
    # DB에 저장된 질문들 가져오기
    questions = interview.questions.all().order_by('order')
    question_data = [{"order": q.order, "text": q.question_text} for q in questions]

    return Response({
        "status": "success",
        "interview_id": interview.id,
        "questions": question_data,
        "report": BehaviorReportSerializer(report).data
    }, status=status.HTTP_200_OK)


# [3.2] 누적 점수 추이 조회
@extend_schema(
    summary="[3.2] 누적 점수 추이 조회",
    responses={200: CumulativeReportSerializer(many=True)},
    tags=['3. Report Flow']
)
@api_view(['GET'])
def get_cumulative_report(request):
    # interview 모델의 created_at을 참조하여 정렬
    reports = BehaviorReport.objects.select_related('interview').order_by('interview__created_at')
    serializer = CumulativeReportSerializer(reports, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)