import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# MediaPipe 랜드마크 인덱스 정의
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
NOSE_TIP = 1         
LEFT_EYE_PUPIL = 468  


def get_pixel_coords(landmark, width, height):
    """랜드마크의 정규화된 좌표를 실제 픽셀 좌표로 변환"""
    return int(landmark.x * width), int(landmark.y * height)


def calculate_distance(p1, p2):
    """두 점 사이의 유클리디안 거리 계산"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_frame_metrics(landmarks, width, height):
    """한 프레임에서 눈, 고개, 시선 관련 메트릭 추출"""
    l_top = get_pixel_coords(landmarks[LEFT_EYE_TOP], width, height)
    l_bottom = get_pixel_coords(landmarks[LEFT_EYE_BOTTOM], width, height)
    l_left = get_pixel_coords(landmarks[LEFT_EYE_LEFT], width, height)
    l_right = get_pixel_coords(landmarks[LEFT_EYE_RIGHT], width, height)

    r_top = get_pixel_coords(landmarks[RIGHT_EYE_TOP], width, height)
    r_bottom = get_pixel_coords(landmarks[RIGHT_EYE_BOTTOM], width, height)
    r_left = get_pixel_coords(landmarks[RIGHT_EYE_LEFT], width, height)
    r_right = get_pixel_coords(landmarks[RIGHT_EYE_RIGHT], width, height)

    nose = get_pixel_coords(landmarks[NOSE_TIP], width, height)
    l_pupil = get_pixel_coords(landmarks[LEFT_EYE_PUPIL], width, height)

    # 1. EAR 계산
    l_vert, l_horz = calculate_distance(l_top, l_bottom), calculate_distance(l_left, l_right)
    r_vert, r_horz = calculate_distance(r_top, r_bottom), calculate_distance(r_left, r_right)
    ear = ((l_vert / l_horz if l_horz > 0 else 0) + (r_vert / r_horz if r_horz > 0 else 0)) / 2.0

    # 2. 고개 회전 (Head Turn) 및 기울기 (Head Tilt)
    dist_nose_to_left = calculate_distance(nose, l_left)
    dist_nose_to_right = calculate_distance(nose, r_right)
    total_eye_width = dist_nose_to_left + dist_nose_to_right
    head_turn = dist_nose_to_left / total_eye_width if total_eye_width > 0 else 0.5
    head_tilt = abs(l_left[1] - r_right[1])

    # 3. 시선 비율 (Gaze Ratio)
    l_eye_width = calculate_distance(l_left, l_right)
    l_pupil_offset = calculate_distance(l_left, l_pupil)
    gaze_ratio = l_pupil_offset / l_eye_width if l_eye_width > 0 else 0.5

    return ear, head_turn, head_tilt, gaze_ratio


# ===========================================================================
# [기능 1] 초기 세팅 영상 분석 함수 (Calibration)
# ===========================================================================
def run_calibration(video_path):
    """
    초기 세팅 영상을 분석하여 사용자의 정면 기준값(Threshold)을 리턴합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"초기 세팅 영상을 열 수 없습니다: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ear_list, head_turn_list, head_tilt_list, gaze_list = [], [], [], []

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear, head_turn, head_tilt, gaze_ratio = get_frame_metrics(landmarks, width, height)
                
                ear_list.append(ear)
                head_turn_list.append(head_turn)
                head_tilt_list.append(head_tilt)
                gaze_list.append(gaze_ratio)
                
    cap.release()

    if not ear_list:
        raise RuntimeError("초기 세팅 영상에서 얼굴을 감지하지 못했습니다.")

    # 상위 80%의 눈뜬 상태 데이터를 기준으로 평균 EAR 계산
    ear_list.sort()
    normal_ear = sum(ear_list[int(len(ear_list)*0.2):]) / len(ear_list[int(len(ear_list)*0.2):])
    
    return {
        "ear_threshold": normal_ear * 0.75,
        "base_head_turn": sum(head_turn_list) / len(head_turn_list),
        "head_turn_tolerance": 0.07,
        "head_tilt_tolerance": (sum(head_tilt_list) / len(head_tilt_list)) + 12,
        "base_gaze_ratio": sum(gaze_list) / len(gaze_list),
        "gaze_tolerance": 0.05
    }


# ===========================================================================
# [기능 2] 본 면접 영상 분석 함수 (기존 구조 유지 + 초기값 반영)
# ===========================================================================
def analyze_behavior_video(video_path, config):
    """
    인자로 넘겨받은 config(초기 세팅값)를 기반으로 본 면접 영상을 정밀 분석합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"면접 영상을 열 수 없습니다: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MediaPipe 초기화
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    pose = mp_pose.Pose()

    # 저장 및 카운트용 변수
    frame_details = []
    blink_count = 0
    is_eye_closed_prev = False

    # 흔들림 측정용 윈도우 설정
    window_size = int(fps)  # 1초 크기의 window
    center_x_window = deque(maxlen=window_size)
    prev_sway_state = False
    lr_sway_count = 0

    # 샘플링 간격 (초당 5프레임 저장)
    sample_interval = max(1, int(fps / 5))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 추론
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        # 기본 detail 딕셔너리 구조 (기존 Django 구조 유지)
        detail = {
            'timestamp': round(frame_idx / fps, 2),
            'is_blink': False,
            'ear_value': 0.0,
            'gaze_direction': 'center',  # 'center' 또는 'deviated'
            'head_turn': 0.0,
            'head_tilt': 0.0,
            'gaze_ratio': 0.0,
            'shoulder_tilt': 0.0,
            'shoulder_width': 0.0,
            'is_swaying': False,
            'smile_ratio': 0.0,
            'is_smiling': False,
        }

        # ------------------------------------------
        # 1. 얼굴 및 시선 분석 (초기값 기준 판정)
        # ------------------------------------------
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            # 테스트 코드의 정밀 메트릭 함수 활용
            ear, head_turn, head_tilt, gaze_ratio = get_frame_metrics(landmarks, width, height)
            
            detail['ear_value'] = float(ear)
            detail['head_turn'] = float(head_turn)
            detail['head_tilt'] = float(head_tilt)
            detail['gaze_ratio'] = float(gaze_ratio)

            # [수정 로직] 초기 세팅값(config) 기준 눈 깜빡임 판정
            is_closed = ear < config["ear_threshold"]
            if is_eye_closed_prev and not is_closed:
                blink_count += 1
                detail['is_blink'] = True
            is_eye_closed_prev = is_closed

            # [수정 로직] 초기 세팅값(config) 기준 정면(center) vs 이탈(deviated) 판정
            is_head_turn_ok = abs(head_turn - config["base_head_turn"]) < config["head_turn_tolerance"]
            is_head_tilt_ok = head_tilt < config["head_tilt_tolerance"]
            is_iris_front = abs(gaze_ratio - config["base_gaze_ratio"]) < config["gaze_tolerance"]
            
            if is_head_turn_ok and is_head_tilt_ok and is_iris_front:
                detail['gaze_direction'] = 'center'
            else:
                detail['gaze_direction'] = 'deviated'

            # 미소 분석 (기존 로직 유지)
            mouth_width = abs(landmarks[61].x - landmarks[291].x)
            eye_width = abs(landmarks[33].x - landmarks[263].x)
            smile_ratio = mouth_width / eye_width if eye_width > 0 else 0
            
            detail['smile_ratio'] = float(smile_ratio)
            detail['is_smiling'] = smile_ratio > 1.7

        # ------------------------------------------
        # 2. 자세 분석 (어깨 및 흔들림 - 기존 로직 유지)
        # ------------------------------------------
        if pose_results.pose_landmarks:
            ps_lm = pose_results.pose_landmarks.landmark
            l_sh = ps_lm[11]
            r_sh = ps_lm[12]

            curr_tilt = l_sh.y - r_sh.y
            curr_width = abs(l_sh.x - r_sh.x)
            curr_center_x = (l_sh.x + r_sh.x) / 2

            detail['shoulder_tilt'] = float(curr_tilt)
            detail['shoulder_width'] = float(curr_width)

            # 1초 윈도우 기준 몸 흔들림 감지
            center_x_window.append(curr_center_x)
            if len(center_x_window) == window_size:
                sway_std = np.std(center_x_window)
                is_sway_now = sway_std > 0.005
                detail['is_swaying'] = is_sway_now

                if is_sway_now and not prev_sway_state:
                    lr_sway_count += 1
                prev_sway_state = is_sway_now

        # 샘플링 주기마다 기록 저장
        if frame_idx % sample_interval == 0:
            frame_details.append(detail)

        frame_idx += 1

    cap.release()

    # ======================================================
    # 3. SUMMARY 요약 데이터 생성 (기존 구조 맞춤)
    # ======================================================
    total_frames = len(frame_details)
    duration_sec = frame_idx / fps
    duration_min = duration_sec / 60

    focus_rate = (
        sum(1 for d in frame_details if d['gaze_direction'] == 'center') / total_frames * 100
        if total_frames > 0 else 0
    )
    
    # 정면이 아니면 모두 시선 이탈(deviated) 처리
    deviated_rate = 100.0 - focus_rate

    smile_rate = (
        sum(1 for d in frame_details if d.get('is_smiling')) / total_frames * 100
        if total_frames > 0 else 0
    )

    shoulder_stability = (
        sum(1 for d in frame_details if abs(d['shoulder_tilt']) < 0.02) / total_frames * 100
        if total_frames > 0 else 0
    )

    summary = {
        'focus_rate': round(focus_rate, 1),
        'deviated_gaze_rate': round(deviated_rate, 1),  # 정면 이탈율로 명칭 변경
        'blink_count': blink_count,
        'nod_count': 0,
        'shoulder_stability': round(shoulder_stability, 1),
        'lr_sway_count': lr_sway_count,
        'fb_sway_count': 0,
        'total_smile_rate': round(smile_rate, 1),

        'duration_sec': round(duration_sec, 2),
    }

    return frame_details, summary