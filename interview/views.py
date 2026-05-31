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

from .models import Interview, Resume
from .serializers import (
    InterviewCreateSerializer,
    InterviewResponseSerializer,
    InterviewListElementSerializer,
    FinalReportSerializer,
    CumulativeTrendsResponseSerializer,
    ResumeListSerializer,
    ResumeDetailSerializer,
    DeleteResponseSerializer,
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
        '새 자소서 작성',
        value={
            "interview_type": "RESUME",
            "resume_title": "프론트엔드 자소서",
            "resume_content": "안녕하세요...",
            "question_count": 5,
            "video_file": "(파일)"
        },
        request_only=True
    ),

    OpenApiExample(
        '기존 자소서 선택',
        value={
            "interview_type": "RESUME",
            "resume_id": 1,
            "question_count": 5,
            "video_file": "(파일)"
        },
        request_only=True
    ),

    OpenApiExample(
        '직무 기반 면접',
        value={
            "interview_type": "JOB",
            "job_category": "Backend",
            "question_count": 5,
            "video_file": "(파일)"
        },
        request_only=True
        ),
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

        # 자소서 처리
        resume = None

        if validated_data['interview_type'] == 'RESUME':

            resume_id = validated_data.get('resume_id')

            if resume_id:

                resume = get_object_or_404(
                    Resume,
                    id=resume_id
                )

            else:

                resume_title = validated_data.get(
                    'resume_title'
                )

                resume_content = validated_data.get(
                    'resume_content'
                )

                if not resume_title or not resume_content:

                    return Response(
                        {
                            "error": "새 자소서 생성 시 resume_title, resume_content는 필수입니다."
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )

                resume = Resume.objects.create(
                    title=resume_title,
                    content=resume_content
                )


        # 면접 생성
        interview = Interview.objects.create(
            interview_type=validated_data['interview_type'],
            resume=resume,
            job_category=validated_data.get(
                'job_category',
                ''
            ),
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
                        "avg_silence_duration": 4.2,
                        "transcript": "안녕하세요 저는 백엔드 개발자 지망생입니다.\n저의 강점은 문제 해결 능력이라고 생각합니다."
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
        transcripts = []
        for q in completed_questions:
            try:
                transcripts.append(q.speech_analysis.report.transcript)
            except Exception:
                pass
        speech_data = {
            "avg_spm": sr.avg_spm,
            "pace": sr.pace,
            "avg_db": sr.avg_db,
            "volume_level": sr.volume_level,
            "total_filler_count": sr.total_filler_count,
            "frequent_fillers": sr.frequent_fillers,
            "total_silence_count": sr.total_silence_count,
            "avg_silence_duration": sr.avg_silence_duration,
            "transcript": "\n".join(transcripts),
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
# ===========================================================================
# 5. 회차별 변화 트렌드 데이터 조회 (GET)
# Endpoint: /interviews/trends/
# ===========================================================================
@extend_schema(
    summary="[5] 회차별 변화 트렌드 데이터 조회",
    description="과거 역대 면접들의 핵심 분석 수치들을 시계열 형태로 반환합니다.",
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
                        "gaze_deviation_ratio": 27.9,
                        "body_sway_per_min": 5.4,
                        "shoulder_stability": 88.3,
                        "blink_per_min": 22.1,
                        "nod_per_min": 3.2,
                        "smile_ratio": 5.0,
                        "avg_spm": 340.2,
                        "pace": "빠름",
                        "avg_db": -20.1,
                        "volume_level": "보통",
                        "total_filler_count": 10,
                        "total_silence_count": 4
                    },
                    {
                        "interview_id": 2,
                        "date": "2026-05-25",
                        "interview_type": "RESUME",
                        "gaze_front_ratio": 85.5,
                        "gaze_deviation_ratio": 14.5,
                        "body_sway_per_min": 2.3,
                        "shoulder_stability": 92.5,
                        "blink_per_min": 15.2,
                        "nod_per_min": 4.1,
                        "smile_ratio": 12.5,
                        "avg_spm": 312.4,
                        "pace": "보통",
                        "avg_db": -18.3,
                        "volume_level": "보통",
                        "total_filler_count": 7,
                        "total_silence_count": 3
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

    completed_interviews = Interview.objects.filter(
        status='COMPLETED'
    ).order_by('created_at')

    trends_list = []

    for interview in completed_interviews:

        completed_questions = interview.questions.filter(
            status='COMPLETED'
        )

        if not completed_questions.exists():
            continue

        # =========================
        # 총 영상 시간
        # =========================
        total_duration = (
            completed_questions.aggregate(
                Sum('video_duration')
            )['video_duration__sum'] or 0.0
        )

        total_duration_min = (
            total_duration / 60
            if total_duration > 0 else 1.0
        )

        # =========================
        # 평균 비율 데이터
        # =========================
        avg_gaze_front = (
            completed_questions.aggregate(
                Avg('gaze_front_ratio')
            )['gaze_front_ratio__avg'] or 0.0
        )

        avg_gaze_dev = (
            completed_questions.aggregate(
                Avg('gaze_deviation_ratio')
            )['gaze_deviation_ratio__avg'] or 0.0
        )

        avg_shoulder = (
            completed_questions.aggregate(
                Avg('shoulder_stability_ratio')
            )['shoulder_stability_ratio__avg'] or 0.0
        )

        avg_smile = (
            completed_questions.aggregate(
                Avg('smile_ratio')
            )['smile_ratio__avg'] or 0.0
        )

        # =========================
        # raw count 합산
        # =========================
        total_sways = (
            completed_questions.aggregate(
                Sum('body_sway_count')
            )['body_sway_count__sum'] or 0
        )

        total_blinks = (
            completed_questions.aggregate(
                Sum('blink_count')
            )['blink_count__sum'] or 0
        )

        total_nods = (
            completed_questions.aggregate(
                Sum('nod_count')
            )['nod_count__sum'] or 0
        )

        # =========================
        # 분당 변환
        # =========================
        body_sway_per_min = total_sways / total_duration_min
        blink_per_min = total_blinks / total_duration_min
        nod_per_min = total_nods / total_duration_min

        # =========================
        # speech 데이터
        # =========================
        speech_data = {}
        if hasattr(interview, 'speech_report'):
            sr = interview.speech_report
            speech_data = {
                "avg_spm": sr.avg_spm,
                "pace": sr.pace,
                "avg_db": sr.avg_db,
                "volume_level": sr.volume_level,
                "total_filler_count": sr.total_filler_count,
                "total_silence_count": sr.total_silence_count,
            }

        # =========================
        # trends row 생성
        # =========================
        trends_list.append({
            "interview_id": interview.id,
            "date": interview.created_at.strftime("%Y-%m-%d"),
            "interview_type": interview.interview_type,

            "gaze_front_ratio": round(avg_gaze_front, 1),
            "gaze_deviation_ratio": round(avg_gaze_dev, 1),

            "body_sway_per_min": round(body_sway_per_min, 1),
            "shoulder_stability": round(avg_shoulder, 1),

            "blink_per_min": round(blink_per_min, 1),
            "nod_per_min": round(nod_per_min, 1),

            "smile_ratio": round(avg_smile, 1),

            **speech_data
        })

    return Response({
        "total_interview_count": len(trends_list),
        "trends": trends_list
    }, status=status.HTTP_200_OK)

@extend_schema(
    summary="[6] 저장된 자소서 목록 조회",
    description="사용자가 저장한 자소서 목록을 조회합니다.",
    responses={200: ResumeListSerializer(many=True)}, 
    examples=[
        OpenApiExample(
            '자소서 목록',
            value=[
                {
                    "resume_id": 1,
                    "title": "프론트엔드 자소서",
                    "created_at": "2026-05-31"
                },
                {
                    "resume_id": 2,
                    "title": "신입 자소서",
                    "created_at": "2026-05-30"
                }
            ],
            response_only=True
        )
    ],
    tags=['2. Resumes']
)
@api_view(['GET'])
def get_resume_list(request):

    resumes = Resume.objects.all().order_by('-created_at')

    result = []

    for resume in resumes:
        result.append({
            "resume_id": resume.id,
            "title": resume.title,
            "created_at": resume.created_at.strftime("%Y-%m-%d")
        })

    return Response(
        result,
        status=status.HTTP_200_OK
    )


@extend_schema(
    summary="[7] 자소서 상세 조회",
    description="특정 자소서 내용을 조회합니다.",
    responses={200: ResumeDetailSerializer},
    examples=[
        OpenApiExample(
            '자소서 상세',
            value={
                "resume_id": 1,
                "title": "프론트엔드 자소서",
                "content": "안녕하세요. 프론트엔드 개발자를 희망하는...",
                "created_at": "2026-05-31"
            },
            response_only=True
        )
    ],
    tags=['2. Resumes']
)
@api_view(['GET'])
def get_resume_detail(request, resume_id):

    resume = get_object_or_404(
        Resume,
        id=resume_id
    )

    return Response({
        "resume_id": resume.id,
        "title": resume.title,
        "content": resume.content,
        "created_at": resume.created_at.strftime("%Y-%m-%d")
    })

@extend_schema(
    summary="[8] 자소서 삭제",
    description="저장된 자소서를 삭제합니다.",
    responses={200: DeleteResponseSerializer},
    examples=[
    OpenApiExample(
        '삭제 성공',
        value={
            "message": "자소서가 삭제되었습니다.",
            "resume_id": 1
        },
        response_only=True
    )
],
    tags=['2. Resumes']
)
@api_view(['DELETE'])
def delete_resume(request, resume_id):

    resume = get_object_or_404(
        Resume,
        id=resume_id
    )

    resume.delete()

    return Response({
        "message": "자소서가 삭제되었습니다.",
        "resume_id": resume_id
    })

@extend_schema(
    summary="[9] 면접 리포트 삭제",
    description="특정 면접 리포트를 삭제합니다.",
    responses={200: DeleteResponseSerializer},
    examples=[
        OpenApiExample(
            '삭제 성공',
            value={
                "message": "면접 리포트가 삭제되었습니다."
            },
            response_only=True
        )
    ],
    tags=['1. Interviews']
)
@api_view(['DELETE'])
def delete_interview_report(
    request,
    interview_id
):

    interview = get_object_or_404(
        Interview,
        id=interview_id
    )

    interview.delete()

    return Response({
        "message": "면접 리포트가 삭제되었습니다.",
        "interview_id": interview_id
    })