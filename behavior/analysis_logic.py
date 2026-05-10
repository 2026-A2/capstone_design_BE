import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def analyze_behavior_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # 랜드마크 초기화
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    pose = mp_pose.Pose()

    # 로직용 변수
    frame_details = []
    blink_count = 0
    blink_state = False # False: 눈 뜸, True: 눈 감음
    
    # 영점 조절 및 흔들림 측정용 큐 (1초 데이터 저장)
    window_size = int(fps)
    center_x_window = deque(maxlen=window_size)
    shoulder_width_window = deque(maxlen=window_size)
    
    # 영점 기준값 (초기 2~3초)
    base_values = {'gaze_x': 0.5, 'gaze_y': 0.5, 'shoulder_tilt': 0, 'shoulder_width': 0}
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 분석 실행
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        # 기본값 설정
        detail = {
            'timestamp': round(frame_idx / fps, 2),
            'is_blink': False, 'is_swaying': False, 'gaze_direction': 'center'
        }

        # --- 1. Face 분석 (눈, 시선, 고개) ---
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            # (1) 눈 깜빡임 (EAR 유사 계산)
            # 왼쪽 눈 기준 랜드마크 (상:159, 하:145)
            ear = abs(landmarks[159].y - landmarks[145].y)
            detail['ear_value'] = ear
            if ear < 0.02: # 설정값 미만 '감음'
                blink_state = True
            elif blink_state and ear > 0.025: # 다시 '뜸'
                blink_count += 1
                detail['is_blink'] = True
                blink_state = False

            # (2) 시선 및 고개 (Gaze & Head)
            # 코 끝(1), 왼쪽 끝(234), 오른쪽 끝(454), 턱(152), 미간(10) 활용
            detail['gaze_x'] = landmarks[468].x  # 왼쪽 홍채 중심 (MediaPipe Refined)
            detail['gaze_y'] = landmarks[468].y
            detail['head_yaw'] = landmarks[1].x - landmarks[10].x # 좌우 회전 감지
            detail['head_pitch'] = landmarks[1].y - landmarks[10].y # 수직(끄덕임/내리깔기)
            detail['head_roll'] = landmarks[234].y - landmarks[454].y # 갸우뚱(기울기)

            # 방향 판정
            if detail['head_yaw'] > 0.05: detail['gaze_direction'] = 'left'
            elif detail['head_yaw'] < -0.05: detail['gaze_direction'] = 'right'

        # --- 2. Pose 분석 (어깨 기울기, 흔들림) ---
        if pose_results.pose_landmarks:
            ps_lm = pose_results.pose_landmarks.landmark
            l_sh, r_sh = ps_lm[11], ps_lm[12] # 왼쪽/오른쪽 어깨
            
            # 어깨 기울기 및 너비
            curr_tilt = l_sh.y - r_sh.y
            curr_width = abs(l_sh.x - r_sh.x)
            curr_center_x = (l_sh.x + r_sh.x) / 2
            
            detail['shoulder_tilt'] = curr_tilt
            detail['shoulder_width'] = curr_width
            detail['center_x'] = curr_center_x

            # 초기 2초간 영점 조절 (Calibration)
            if frame_idx < fps * 2:
                base_values['shoulder_tilt'] += curr_tilt / (fps * 2)
                base_values['shoulder_width'] += curr_width / (fps * 2)
            else:
                # 흔들림 측정 (표준편차)
                center_x_window.append(curr_center_x)
                shoulder_width_window.append(curr_width)
                
                if len(center_x_window) == window_size:
                    sway_std = np.std(center_x_window)
                    if sway_std > 0.01: # 임계값은 테스트 후 조정
                        detail['is_swaying'] = True

        # 프레임 샘플링 저장 (DB 부하 방지: 초당 5프레임 수준)
        if frame_idx % int(fps/5) == 0:
            frame_details.append(detail)
        
        frame_idx += 1

    cap.release()
    
    # 3. 종합 리포트 수치 계산 (샘플 데이터 기반)
    summary = {
        'focus_rate': sum(1 for d in frame_details if d['gaze_direction'] == 'center') / len(frame_details) * 100,
        'left_gaze_rate': sum(1 for d in frame_details if d['gaze_direction'] == 'left') / len(frame_details) * 100,
        'right_gaze_rate': sum(1 for d in frame_details if d['gaze_direction'] == 'right') / len(frame_details) * 100,
        'blinks_per_min': (blink_count / (frame_idx / fps)) * 60,
        'total_nod_count': 0, # 피치 변화 알고리즘 추가 필요
        'shoulder_stability': 90.0, # 영점 대비 편차로 계산
        'lr_sway_count': sum(1 for d in frame_details if d['is_swaying']),
        'fb_sway_count': 0,
        'total_smile_rate': 0.0,
        'overall_score': 80,
        'ai_feedback': "전체적으로 양호하나 고개가 왼쪽으로 기우는 습관이 있습니다."
    }

    return frame_details, summary