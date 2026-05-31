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
        base_options=mp_python.BaseOptions(
            model_asset_path=_FACE_MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    return vision.FaceLandmarker.create_from_options(options)


def _make_pose_landmarker():
    _ensure_models()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=_POSE_MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
    )

    return vision.PoseLandmarker.create_from_options(options)


# ==================================================
# Landmark
# ==================================================

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

NOSE_TIP = 1
LEFT_EYE_PUPIL = 468

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14


# ==================================================
# Utility
# ==================================================

def get_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def get_np_coords(landmark, width, height):
    return np.array([
        landmark.x * width,
        landmark.y * height
    ], dtype=np.float32)


def calculate_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )


def calculate_np_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


# ==================================================
# 기존 시선 / 눈깜빡임 지표
# ==================================================

def get_frame_metrics(landmarks, width, height):
    l_top = get_pixel_coords(
        landmarks[LEFT_EYE_TOP],
        width,
        height
    )
    l_bottom = get_pixel_coords(
        landmarks[LEFT_EYE_BOTTOM],
        width,
        height
    )
    l_left = get_pixel_coords(
        landmarks[LEFT_EYE_LEFT],
        width,
        height
    )
    l_right = get_pixel_coords(
        landmarks[LEFT_EYE_RIGHT],
        width,
        height
    )

    r_top = get_pixel_coords(
        landmarks[RIGHT_EYE_TOP],
        width,
        height
    )
    r_bottom = get_pixel_coords(
        landmarks[RIGHT_EYE_BOTTOM],
        width,
        height
    )
    r_left = get_pixel_coords(
        landmarks[RIGHT_EYE_LEFT],
        width,
        height
    )
    r_right = get_pixel_coords(
        landmarks[RIGHT_EYE_RIGHT],
        width,
        height
    )

    nose = get_pixel_coords(
        landmarks[NOSE_TIP],
        width,
        height
    )
    l_pupil = get_pixel_coords(
        landmarks[LEFT_EYE_PUPIL],
        width,
        height
    )

    l_vert = calculate_distance(l_top, l_bottom)
    l_horz = calculate_distance(l_left, l_right)

    r_vert = calculate_distance(r_top, r_bottom)
    r_horz = calculate_distance(r_left, r_right)

    l_ear = l_vert / l_horz if l_horz > 0 else 0
    r_ear = r_vert / r_horz if r_horz > 0 else 0

    ear = (l_ear + r_ear) / 2.0

    dist_nose_to_left = calculate_distance(nose, l_left)
    dist_nose_to_right = calculate_distance(nose, r_right)

    total_eye_width = dist_nose_to_left + dist_nose_to_right

    head_turn = (
        dist_nose_to_left / total_eye_width
        if total_eye_width > 0
        else 0.5
    )

    head_tilt = abs(l_left[1] - r_right[1])

    l_eye_width = calculate_distance(l_left, l_right)
    l_pupil_offset = calculate_distance(l_left, l_pupil)

    gaze_ratio = (
        l_pupil_offset / l_eye_width
        if l_eye_width > 0
        else 0.5
    )

    return ear, head_turn, head_tilt, gaze_ratio


# ==================================================
# 미소 / 끄덕임용 지표
# ==================================================

def get_smile_nod_features(landmarks, width, height):
    p_nose = get_np_coords(
        landmarks[NOSE_TIP],
        width,
        height
    )

    p_left_eye = get_np_coords(
        landmarks[LEFT_EYE_LEFT],
        width,
        height
    )

    p_right_eye = get_np_coords(
        landmarks[RIGHT_EYE_RIGHT],
        width,
        height
    )

    p_mouth_left = get_np_coords(
        landmarks[MOUTH_LEFT],
        width,
        height
    )

    p_mouth_right = get_np_coords(
        landmarks[MOUTH_RIGHT],
        width,
        height
    )

    p_upper_lip = get_np_coords(
        landmarks[UPPER_LIP],
        width,
        height
    )

    p_lower_lip = get_np_coords(
        landmarks[LOWER_LIP],
        width,
        height
    )

    eye_center = (
        p_left_eye + p_right_eye
    ) / 2.0

    face_scale = calculate_np_distance(
        p_left_eye,
        p_right_eye
    )

    if face_scale <= 0:
        face_scale = 1.0

    nose_pitch_ratio = (
        p_nose[1] - eye_center[1]
    ) / face_scale

    mouth_width = calculate_np_distance(
        p_mouth_left,
        p_mouth_right
    )

    mouth_height = calculate_np_distance(
        p_upper_lip,
        p_lower_lip
    )

    if mouth_height <= 0:
        mouth_height = 1.0

    mouth_ratio = mouth_width / mouth_height

    left_corner_raise = (
        p_nose[1] - p_mouth_left[1]
    ) / face_scale

    right_corner_raise = (
        p_nose[1] - p_mouth_right[1]
    ) / face_scale

    corner_raise = (
        left_corner_raise
        + right_corner_raise
    ) / 2.0

    return {
        "nose_pitch_ratio": float(nose_pitch_ratio),
        "mouth_ratio": float(mouth_ratio),
        "corner_raise": float(corner_raise),
    }


# ==================================================
# Calibration
# ==================================================

def run_calibration(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"초기 세팅 영상을 열 수 없습니다: {video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ear_list = []
    head_turn_list = []
    head_tilt_list = []
    gaze_list = []

    pitch_ratios = []
    mouth_ratios = []
    corner_raises = []

    with _make_face_landmarker() as face_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = face_landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                ear, head_turn, head_tilt, gaze_ratio = (
                    get_frame_metrics(
                        landmarks,
                        width,
                        height
                    )
                )

                ear_list.append(ear)
                head_turn_list.append(head_turn)
                head_tilt_list.append(head_tilt)
                gaze_list.append(gaze_ratio)

                features = get_smile_nod_features(
                    landmarks,
                    width,
                    height
                )

                pitch_ratios.append(
                    features["nose_pitch_ratio"]
                )
                mouth_ratios.append(
                    features["mouth_ratio"]
                )
                corner_raises.append(
                    features["corner_raise"]
                )

    cap.release()

    if not ear_list:
        raise RuntimeError(
            "초기 세팅 영상에서 얼굴을 감지하지 못했습니다."
        )

    ear_list.sort()

    start_idx = int(len(ear_list) * 0.2)
    valid_ear_list = ear_list[start_idx:]

    if not valid_ear_list:
        valid_ear_list = ear_list

    normal_ear = sum(valid_ear_list) / len(valid_ear_list)

    return {
        "ear_threshold": float(normal_ear * 0.75),

        "base_head_turn": float(
            sum(head_turn_list) / len(head_turn_list)
        ),
        "head_turn_tolerance": 0.07,

        "head_tilt_tolerance": float(
            (sum(head_tilt_list) / len(head_tilt_list)) + 12
        ),

        "base_gaze_ratio": float(
            sum(gaze_list) / len(gaze_list)
        ),
        "gaze_tolerance": 0.05,

        "base_pitch_ratio": float(
            np.mean(pitch_ratios)
        ),
        "base_mouth_ratio": float(
            np.mean(mouth_ratios)
        ),
        "base_corner_raise": float(
            np.mean(corner_raises)
        ),
    }


# ==================================================
# Analysis
# ==================================================

def analyze_behavior_video(video_path, config):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"면접 영상을 열 수 없습니다: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    #동영상 시간 체크용 fps 
    print(fps)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_details = []

    blink_count = 0
    is_eye_closed_prev = False

    window_size = max(1, int(fps))
    center_x_window = deque(maxlen=window_size)

    prev_sway_state = False
    lr_sway_count = 0

    sample_interval = max(1, int(fps / 5))
    frame_idx = 0

    # ==================================================
    # Smile
    # ==================================================

    smile_score_accumulator = 0.0
    analyzed_face_frame_count = 0

    score_buffer = deque(
        maxlen=max(1, int(fps * 0.3))
    )

    # ==================================================
    # Nod
    # ==================================================

    nod_count = 0
    nod_state = "IDLE"

    smooth_nod_delta = None
    prev_smooth_nod_delta = None

    nod_peak_delta = 0.0
    last_nod_frame = -999999

    EMA_ALPHA = 0.5
    START_DOWN_DELTA = 0.015
    MIN_NOD_DEPTH = 0.035
    VELOCITY_THRESHOLD = 0.002
    NOD_COOLDOWN = max(1, int(fps * 0.15))

    required_keys = [
        "ear_threshold",
        "base_head_turn",
        "head_turn_tolerance",
        "head_tilt_tolerance",
        "base_gaze_ratio",
        "gaze_tolerance",
        "base_pitch_ratio",
        "base_mouth_ratio",
        "base_corner_raise",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise RuntimeError(
            f"캘리브레이션 설정값이 누락되었습니다: {missing_keys}"
        )

    with _make_face_landmarker() as face_landmarker, _make_pose_landmarker() as pose_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            detail = {
                "timestamp": round(frame_idx / fps, 2),

                "is_blink": False,
                "ear_value": 0.0,

                "gaze_direction": "center",
                "head_turn": 0.0,
                "head_tilt": 0.0,
                "gaze_ratio": 0.0,

                "shoulder_tilt": 0.0,
                "shoulder_width": 0.0,
                "is_swaying": False,
                "shoulder_stable": True,

                "smile_ratio": 0.0,
                "is_smiling": False,

                "is_nodding": False,
                "nod_delta": 0.0,
            }

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]

                ear, head_turn, head_tilt, gaze_ratio = (
                    get_frame_metrics(
                        landmarks,
                        width,
                        height
                    )
                )

                detail["ear_value"] = float(ear)
                detail["head_turn"] = float(head_turn)
                detail["head_tilt"] = float(head_tilt)
                detail["gaze_ratio"] = float(gaze_ratio)

                # ==================================================
                # Blink
                # ==================================================

                is_closed = ear < config["ear_threshold"]

                if is_eye_closed_prev and not is_closed:
                    blink_count += 1
                    detail["is_blink"] = True

                is_eye_closed_prev = is_closed

                # ==================================================
                # Gaze
                # ==================================================

                is_head_turn_ok = (
                    abs(
                        head_turn
                        - config["base_head_turn"]
                    )
                    < config["head_turn_tolerance"]
                )

                is_head_tilt_ok = (
                    head_tilt
                    < config["head_tilt_tolerance"]
                )

                is_iris_front = (
                    abs(
                        gaze_ratio
                        - config["base_gaze_ratio"]
                    )
                    < config["gaze_tolerance"]
                )

                if (
                    is_head_turn_ok
                    and is_head_tilt_ok
                    and is_iris_front
                ):
                    detail["gaze_direction"] = "center"
                else:
                    detail["gaze_direction"] = "deviated"

                # ==================================================
                # Smile
                # ==================================================

                features = get_smile_nod_features(
                    landmarks,
                    width,
                    height
                )

                analyzed_face_frame_count += 1

                ratio_delta = (
                    features["mouth_ratio"]
                    - config["base_mouth_ratio"]
                )

                corner_delta = (
                    features["corner_raise"]
                    - config["base_corner_raise"]
                )

                smile_score = (
                    ratio_delta * 0.1
                    + corner_delta * 0.9
                )

                score_buffer.append(smile_score)

                smooth_score = float(
                    np.mean(score_buffer)
                )

                detail["smile_ratio"] = float(
                    max(0.0, smooth_score)
                )

                if (
                    corner_delta > 0.025
                    and smooth_score > 0.018
                ):
                    detail["is_smiling"] = True
                    smile_score_accumulator += 1.0

                elif (
                    corner_delta > 0.008
                    and smooth_score > 0.005
                ):
                    detail["is_smiling"] = True
                    smile_score_accumulator += 0.7

                elif corner_delta > -0.01:
                    detail["is_smiling"] = True
                    smile_score_accumulator += 0.3

                else:
                    detail["is_smiling"] = False

                # ==================================================
                # Nod
                # ==================================================

                raw_nod_delta = (
                    features["nose_pitch_ratio"]
                    - config["base_pitch_ratio"]
                )

                detail["nod_delta"] = float(raw_nod_delta)

                if smooth_nod_delta is None:
                    smooth_nod_delta = raw_nod_delta
                else:
                    smooth_nod_delta = (
                        EMA_ALPHA * raw_nod_delta
                        + (1 - EMA_ALPHA) * smooth_nod_delta
                    )

                velocity = 0.0

                if prev_smooth_nod_delta is not None:
                    velocity = (
                        smooth_nod_delta
                        - prev_smooth_nod_delta
                    )

                if nod_state == "IDLE":
                    if smooth_nod_delta > START_DOWN_DELTA:
                        nod_state = "DOWN"
                        nod_peak_delta = smooth_nod_delta

                elif nod_state == "DOWN":
                    if smooth_nod_delta > nod_peak_delta:
                        nod_peak_delta = smooth_nod_delta

                    if (
                        nod_peak_delta > MIN_NOD_DEPTH
                        and velocity < -VELOCITY_THRESHOLD
                    ):
                        if (
                            frame_idx - last_nod_frame
                            > NOD_COOLDOWN
                        ):
                            nod_count += 1
                            last_nod_frame = frame_idx
                            detail["is_nodding"] = True

                        nod_state = "IDLE"
                        nod_peak_delta = 0.0

                    elif smooth_nod_delta < 0:
                        nod_state = "IDLE"
                        nod_peak_delta = 0.0

                prev_smooth_nod_delta = smooth_nod_delta

            if pose_result.pose_landmarks:
                ps_lm = pose_result.pose_landmarks[0]

                l_sh = ps_lm[11]
                r_sh = ps_lm[12]

                curr_tilt = l_sh.y - r_sh.y
                curr_width = abs(l_sh.x - r_sh.x)
                curr_center_x = (
                    l_sh.x + r_sh.x
                ) / 2

                detail["shoulder_tilt"] = float(curr_tilt)
                detail["shoulder_width"] = float(curr_width)

                detail["shoulder_stable"] = (
                    abs(curr_tilt) < 0.02
                )

                center_x_window.append(curr_center_x)

                if len(center_x_window) == window_size:
                    sway_std = np.std(center_x_window)
                    is_sway_now = sway_std > 0.005

                    detail["is_swaying"] = bool(is_sway_now)

                    if (
                        is_sway_now
                        and not prev_sway_state
                    ):
                        lr_sway_count += 1

                    prev_sway_state = is_sway_now

            if frame_idx % sample_interval == 0:
                frame_details.append(detail)

            frame_idx += 1

    cap.release()

    total_sampled_frames = len(frame_details)

    duration_sec = (
        frame_idx / fps
        if fps > 0
        else 0.0
    )
    #fps랑 시간 체크용 
    print("=" * 50)
    print("video:", video_path)
    print("fps:", fps)
    print("frame_idx:", frame_idx)
    print("duration_sec:", duration_sec)
    print("=" * 50)

    duration_min = (
        duration_sec / 60
        if duration_sec > 0
        else 1.0
    )

    focus_rate = (
        sum(
            1 for d in frame_details
            if d["gaze_direction"] == "center"
        )
        / total_sampled_frames
        * 100
        if total_sampled_frames > 0
        else 0.0
    )

    deviated_rate = 100.0 - focus_rate

    total_smile_rate = (
        smile_score_accumulator
        / analyzed_face_frame_count
        * 100
        if analyzed_face_frame_count > 0
        else 0.0
    )

    shoulder_stability = (
        sum(
            1 for d in frame_details
            if d.get("shoulder_stable", True)
        )
        / total_sampled_frames
        * 100
        if total_sampled_frames > 0
        else 100.0
    )

    blinks_per_min = (
        blink_count / duration_min
        if duration_min > 0
        else 0.0
    )

    summary = {
        "focus_rate": round(focus_rate, 1),
        "deviated_gaze_rate": round(deviated_rate, 1),

        "blink_count": int(blink_count),
        "blinks_per_min": round(blinks_per_min, 1),

        "nod_count": int(nod_count),

        "shoulder_stability": round(
            shoulder_stability,
            1
        ),

        "lr_sway_count": int(lr_sway_count),
        "fb_sway_count": 0,

        "total_smile_rate": round(
            total_smile_rate,
            1
        ),

        "duration_sec": round(
            duration_sec,
            2
        ),
    }

    return frame_details, summary