from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiExample, inline_serializer 
from rest_framework import serializers

from .models import Interview
from .serializers import (
    InterviewCreateSerializer, 
    InterviewResponseSerializer, 
    BehaviorReportSerializer
)
from behavior.models import BehaviorDetail,BehaviorReport

# [1.1] 면접 세션 생성
@extend_schema(
    summary="[1.1] 면접 세션 생성",
    request=InterviewCreateSerializer, 
    responses={201: InterviewResponseSerializer}, 
    tags=['1. Interview Flow']
)
@api_view(['POST'])
def create_interview(request):
    serializer = InterviewCreateSerializer(data=request.data)
    if serializer.is_valid():
        interview = serializer.save()
        return Response(InterviewResponseSerializer(interview).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# [2.2] 면접 결과 리포트 조회
@extend_schema(
    summary="[2.2] 면접 결과 리포트 조회",
    description="면접이 완료된(completed) 후, 최종 12개 분석 지표를 조회합니다.",
    responses={200: BehaviorReportSerializer},
    examples=[
        OpenApiExample(
            '최종 리포트 응답 예시',
            value={
                "status": "success",
                "interview_id": 1,
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
            response_only=True,
        )
    ],
    tags=['2. Analysis Flow']
)

# interview/views.py

@api_view(['GET']) 
def get_final_report(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id)
    
    # 1. 이미 생성된 리포트가 있는지 확인
    report = BehaviorReport.objects.filter(interview=interview).first()
    
    if not report:
        # 2. 리포트가 없다면 상세 데이터(Detail)를 긁어모음
        all_details = BehaviorDetail.objects.filter(question__interview=interview)
        
        if all_details.exists():
            # [오타 해결 지점] 변수명을 명확하게 정의합니다.
            total_rows = all_details.count() 
            
            # 3. 비율 계산을 위한 분자 값들 추출
            focus_rows = all_details.filter(gaze_direction='center').count()
            left_rows = all_details.filter(gaze_direction='left').count()
            right_rows = all_details.filter(gaze_direction='right').count()
            
            # 4. BehaviorReport 생성 (시리얼라이저 필드와 1:1 매칭)
            report = BehaviorReport.objects.create(
                interview=interview,
                focus_rate=(focus_rows / total_rows) * 100 if total_rows > 0 else 0,
                left_gaze_rate=(left_rows / total_rows) * 100 if total_rows > 0 else 0,
                right_gaze_rate=(right_rows / total_rows) * 100 if total_rows > 0 else 0,
                blinks_per_min=12.0, # 추후 로직 보완 가능
                nod_count=0, 
                shoulder_stability=92.0,
                lr_sway_count=all_details.filter(is_swaying=True).count(),
                fb_sway_count=0,
                total_smile_rate=all_details.filter(is_smiling=True).count() / total_rows * 100 if total_rows > 0 else 0,
                start_smile_status=True,
                end_smile_status=True,
                overall_score=85
            )
        else:
            return Response({
                "status": "not_found",
                "message": "상세 분석 데이터가 없습니다. 영상 업로드가 정상적으로 완료되었는지 확인하세요."
            }, status=status.HTTP_404_NOT_FOUND)

    # 5. 사용자님이 만든 Serializer로 데이터 포장
    serializer = BehaviorReportSerializer(report)
    
    return Response({
        "status": "success",
        "interview_id": interview.id,
        "report": serializer.data
    }, status=status.HTTP_200_OK)


# [3.3] 면접 기록 목록 조회
@extend_schema(
    summary="[3.3] 내 면접 기록 목록 조회",
    responses={200: InterviewResponseSerializer(many=True)},
    tags=['3. Report Flow']
)
@api_view(['GET'])
def list_interviews(request):
    interviews = Interview.objects.all().order_by('-id')
    serializer = InterviewResponseSerializer(interviews, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)