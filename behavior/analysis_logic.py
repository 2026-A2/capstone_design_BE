import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import urllib.request
from collections import deque

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
_FACE_MODEL_PATH = os.path.join(_MODEL_DIR, 'face_landmarker.task')
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, 'pose_landmarker.task')

_FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
_POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'


def _ensure_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_FACE_MODEL_PATH):
        print("face_landmarker.task 모델 다운로드 중...")
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    if not os.path.exists(_POSE_MODEL_PATH):
        print("pose_landmarker.task 모델 다운로드 중...")
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)


def _make_face_landmarker():
    _ensure_models()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _make_pose_landmarker():
    _ensure_models()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


# MediaPipe 랜드마크 인덱스 정의
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
NOSE_TIP = 1
LEFT_EYE_PUPIL = 468


def get_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_frame_metrics(landmarks, width, height):
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

    l_vert, l_horz = calculate_distance(l_top, l_bottom), calculate_distance(l_left, l_right)
    r_vert, r_horz = calculate_distance(r_top, r_bottom), calculate_distance(r_left, r_right)
    ear = ((l_vert / l_horz if l_horz > 0 else 0) + (r_vert / r_horz if r_horz > 0 else 0)) / 2.0

    dist_nose_to_left = calculate_distance(nose, l_left)
    dist_nose_to_right = calculate_distance(nose, r_right)
    total_eye_width = dist_nose_to_left + dist_nose_to_right
    head_turn = dist_nose_to_left / total_eye_width if total_eye_width > 0 else 0.5
    head_tilt = abs(l_left[1] - r_right[1])

    l_eye_width = calculate_distance(l_left, l_right)
    l_pupil_offset = calculate_distance(l_left, l_pupil)
    gaze_ratio = l_pupil_offset / l_eye_width if l_eye_width > 0 else 0.5

    return ear, head_turn, head_tilt, gaze_ratio


def run_calibration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"초기 세팅 영상을 열 수 없습니다: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ear_list, head_turn_list, head_tilt_list, gaze_list = [], [], [], []

    with _make_face_landmarker() as face_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                ear, head_turn, head_tilt, gaze_ratio = get_frame_metrics(landmarks, width, height)
                ear_list.append(ear)
                head_turn_list.append(head_turn)
                head_tilt_list.append(head_tilt)
                gaze_list.append(gaze_ratio)

    cap.release()

    if not ear_list:
        raise RuntimeError("초기 세팅 영상에서 얼굴을 감지하지 못했습니다.")

    ear_list.sort()
    normal_ear = sum(ear_list[int(len(ear_list) * 0.2):]) / len(ear_list[int(len(ear_list) * 0.2):])

    return {
        "ear_threshold": normal_ear * 0.75,
        "base_head_turn": sum(head_turn_list) / len(head_turn_list),
        "head_turn_tolerance": 0.07,
        "head_tilt_tolerance": (sum(head_tilt_list) / len(head_tilt_list)) + 12,
        "base_gaze_ratio": sum(gaze_list) / len(gaze_list),
        "gaze_tolerance": 0.05
    }


def analyze_behavior_video(video_path, config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"면접 영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_details = []
    blink_count = 0
    is_eye_closed_prev = False

    window_size = int(fps)
    center_x_window = deque(maxlen=window_size)
    prev_sway_state = False
    lr_sway_count = 0

    sample_interval = max(1, int(fps / 5))
    frame_idx = 0

    with _make_face_landmarker() as face_landmarker, _make_pose_landmarker() as pose_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            detail = {
                'timestamp': round(frame_idx / fps, 2),
                'is_blink': False,
                'ear_value': 0.0,
                'gaze_direction': 'center',
                'head_turn': 0.0,
                'head_tilt': 0.0,
                'gaze_ratio': 0.0,
                'shoulder_tilt': 0.0,
                'shoulder_width': 0.0,
                'is_swaying': False,
                'smile_ratio': 0.0,
                'is_smiling': False,
            }

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]

                ear, head_turn, head_tilt, gaze_ratio = get_frame_metrics(landmarks, width, height)

                detail['ear_value'] = float(ear)
                detail['head_turn'] = float(head_turn)
                detail['head_tilt'] = float(head_tilt)
                detail['gaze_ratio'] = float(gaze_ratio)

                is_closed = ear < config["ear_threshold"]
                if is_eye_closed_prev and not is_closed:
                    blink_count += 1
                    detail['is_blink'] = True
                is_eye_closed_prev = is_closed

                is_head_turn_ok = abs(head_turn - config["base_head_turn"]) < config["head_turn_tolerance"]
                is_head_tilt_ok = head_tilt < config["head_tilt_tolerance"]
                is_iris_front = abs(gaze_ratio - config["base_gaze_ratio"]) < config["gaze_tolerance"]

                if is_head_turn_ok and is_head_tilt_ok and is_iris_front:
                    detail['gaze_direction'] = 'center'
                else:
                    detail['gaze_direction'] = 'deviated'

                mouth_width = abs(landmarks[61].x - landmarks[291].x)
                eye_width = abs(landmarks[33].x - landmarks[263].x)
                smile_ratio = mouth_width / eye_width if eye_width > 0 else 0

                detail['smile_ratio'] = float(smile_ratio)
                detail['is_smiling'] = smile_ratio > 1.7

            if pose_result.pose_landmarks:
                ps_lm = pose_result.pose_landmarks[0]
                l_sh = ps_lm[11]
                r_sh = ps_lm[12]

                curr_tilt = l_sh.y - r_sh.y
                curr_width = abs(l_sh.x - r_sh.x)
                curr_center_x = (l_sh.x + r_sh.x) / 2

                detail['shoulder_tilt'] = float(curr_tilt)
                detail['shoulder_width'] = float(curr_width)

                center_x_window.append(curr_center_x)
                if len(center_x_window) == window_size:
                    sway_std = np.std(center_x_window)
                    is_sway_now = sway_std > 0.005
                    detail['is_swaying'] = is_sway_now

                    if is_sway_now and not prev_sway_state:
                        lr_sway_count += 1
                    prev_sway_state = is_sway_now

            if frame_idx % sample_interval == 0:
                frame_details.append(detail)

            frame_idx += 1

    cap.release()

    total_frames = len(frame_details)
    duration_sec = frame_idx / fps
    duration_min = duration_sec / 60

    focus_rate = (
        sum(1 for d in frame_details if d['gaze_direction'] == 'center') / total_frames * 100
        if total_frames > 0 else 0
    )
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
        'deviated_gaze_rate': round(deviated_rate, 1),
        'blinks_per_min': round(blink_count / duration_min, 1) if duration_min > 0 else 0,
        'nod_count': 0,
        'shoulder_stability': round(shoulder_stability, 1),
        'lr_sway_count': lr_sway_count,
        'fb_sway_count': 0,
        'total_smile_rate': round(smile_rate, 1),
    }

    return frame_details, summary
