import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import urllib.request
from collections import deque


_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_FACE_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker.task")

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


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
# Landmark Constants
# ==================================================

LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

NOSE_TIP      = 1
LEFT_EYE_PUPIL = 468

MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
UPPER_LIP   = 13
LOWER_LIP   = 14

# 고개 끄덕임용
FOREHEAD   = 10
CHIN       = 152
LEFT_FACE  = 234
RIGHT_FACE = 454

# Pose landmark
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12

RIGHT_EYE_PUPIL = 473


# ==================================================
# 기준값 / 임계값
# ==================================================

# 면접 영상 초반 baseline 구간
BODY_SHOULDER_BASELINE_SEC = 5.0
NOD_BASELINE_SEC           = 3.0

# 너무 짧은 영상일 때 baseline을 3초로 축소
SHORT_VIDEO_THRESHOLD    = 10.0
SHORT_VIDEO_BASELINE_SEC = 3.0

# Pose visibility
MIN_SHOULDER_VISIBILITY = 0.5

# 몸통 흔들림 threshold
LR_SWAY_THRESHOLD   = 0.055
LR_RETURN_THRESHOLD = 0.035

FB_WIDTH_THRESHOLD = 0.040
FB_Y_THRESHOLD     = 0.050
FB_FACE_THRESHOLD  = 0.045

BODY_SMOOTHING_FRAMES = 5

# FB 가중합 점수 방식
FB_SCORE_THRESHOLD        = 0.9
FB_SCORE_RETURN_THRESHOLD = 0.35
FB_WEIGHT_WIDTH           = 0.25
FB_WEIGHT_Y               = 0.25
FB_WEIGHT_FACE            = 0.50

# LR 활성 구간 FB 억제
LR_ACTIVE_RATIO = 0.70

# ==================================================
# [수정] 몸통 흔들림 단일 이벤트 쿨다운
# LR/FB 동시 감지 시 중복 카운트 방지.
# 마지막 흔들림 이벤트 이후 이 시간(초)이 지나야 다음 이벤트로 카운트.
# ==================================================
BODY_SWAY_COOLDOWN_SEC = 0.5

# 어깨 안정성 threshold
SHOULDER_ANGLE_TOLERANCE    = 2.0   # 각도 기반: 기준 어깨각도에서 이 이상 벗어나면 불안정
SHOULDER_POSITION_TOLERANCE = 0.035 # 위치 기반: 어깨 중심이 기준 위치에서 이 비율 이상 이탈
SHOULDER_SMOOTHING_FRAMES   = 3     # 스무딩 줄여 짧은 이탈도 빠르게 반영
REQUIRED_TILT_FRAMES        = 3     # 3프레임(~0.1초) 지속 시 불안정 확정
REQUIRED_STABLE_FRAMES      = 3     # 3프레임 지속 시 안정 복귀

# 고개 끄덕임 threshold
DOWN_THRESHOLD          = 0.040
MIN_NOD_DEPTH           = 0.040
COOLDOWN_SEC            = 0.15
NOD_SMOOTHING_WINDOW    = 4
VELOCITY_WINDOW         = 2
VELOCITY_DOWN_THRESHOLD = 0.0015
VELOCITY_UP_THRESHOLD   = -0.0015
MIN_FACE_HEIGHT         = 80
TILT_WARNING_ANGLE      = 10.0
IDLE_RETURN_RATIO       = 0.5

USE_IGNORE_UP_FILTER      = False
UP_IGNORE_THRESHOLD       = -0.035
BASELINE_RETURN_THRESHOLD = 0.015

# 캘리브레이션 구간에서 중앙값 대비 이 이상 벗어난 프레임 제외
CALIB_REJECT_THRESHOLD = 0.020


# ==================================================
# Utility
# ==================================================

def get_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def get_np_coords(landmark, width, height):
    return np.array([
        landmark.x * width,
        landmark.y * height,
    ], dtype=np.float32)


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_np_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def update_buffer(buffer, value, max_len):
    buffer.append(value)
    if len(buffer) > max_len:
        buffer.pop(0)
    return buffer


def get_median(buffer):
    if not buffer:
        return 0.0
    return float(np.median(np.array(buffer)))


def calculate_shoulder_angle(left_shoulder, right_shoulder):
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    if dx == 0:
        return 90.0
    return math.degrees(math.atan2(dy, dx))


def calculate_fb_score(fb_width_ratio, fb_y_ratio, fb_face_ratio):
    """
    앞뒤 통합 점수 계산.
    각 신호를 임계값으로 정규화 후 가중합.
    양수 = 앞으로 숙임, 음수 = 뒤로 젖힘.
    """
    norm_width = fb_width_ratio / FB_WIDTH_THRESHOLD
    norm_y     = fb_y_ratio     / FB_Y_THRESHOLD
    norm_face  = fb_face_ratio  / FB_FACE_THRESHOLD

    return (
        norm_width * FB_WEIGHT_WIDTH
        + norm_y   * FB_WEIGHT_Y
        + norm_face * FB_WEIGHT_FACE
    )


def get_pose_shoulder_metrics(pose_landmarks, width, height):
    if not pose_landmarks or len(pose_landmarks) <= RIGHT_SHOULDER:
        return None

    left  = pose_landmarks[LEFT_SHOULDER]
    right = pose_landmarks[RIGHT_SHOULDER]

    left_visibility  = getattr(left,  "visibility", 1.0)
    right_visibility = getattr(right, "visibility", 1.0)

    if (
        left_visibility  < MIN_SHOULDER_VISIBILITY
        or right_visibility < MIN_SHOULDER_VISIBILITY
    ):
        return None

    left_px  = np.array([left.x  * width, left.y  * height], dtype=np.float32)
    right_px = np.array([right.x * width, right.y * height], dtype=np.float32)

    center_x = float((left_px[0] + right_px[0]) / 2.0)
    center_y = float((left_px[1] + right_px[1]) / 2.0)
    width_px = float(np.linalg.norm(left_px - right_px))
    # MediaPipe에서 LEFT_SHOULDER는 화면 오른쪽, RIGHT_SHOULDER는 화면 왼쪽에 위치.
    # 화면 왼쪽→오른쪽 방향(right_px→left_px)으로 계산해야 dx>0이 되어
    # atan2가 ±180° 불연속 없이 0° 근처의 안정적인 값을 반환.
    angle    = calculate_shoulder_angle(right_px, left_px)

    return {
        "left_shoulder":  left_px,
        "right_shoulder": right_px,
        "center_x":       center_x,
        "center_y":       center_y,
        "shoulder_width": width_px,
        "shoulder_angle": float(angle),
        "visibility":     float(min(left_visibility, right_visibility)),
    }


def get_face_size_from_landmarks(landmarks, width, height):
    """
    얼굴 크기 변화로 앞뒤 움직임을 보조 판단.
    얼굴 폭과 높이의 평균을 사용.
    """
    if not landmarks or len(landmarks) <= RIGHT_FACE:
        return None

    left_face  = get_np_coords(landmarks[LEFT_FACE],  width, height)
    right_face = get_np_coords(landmarks[RIGHT_FACE], width, height)
    forehead   = get_np_coords(landmarks[FOREHEAD],   width, height)
    chin       = get_np_coords(landmarks[CHIN],       width, height)

    face_width  = calculate_np_distance(left_face,  right_face)
    face_height = calculate_np_distance(forehead, chin)

    if face_width <= 0 or face_height <= 0:
        return None

    return float((face_width + face_height) / 2.0)


def extract_nod_features(landmarks, width, height, min_face_height=MIN_FACE_HEIGHT):
    """
    고개 끄덕임 분석용 feature.
    normalized_nose_y가 커지면 고개를 아래로 숙인 것.
    """
    if not landmarks or len(landmarks) <= RIGHT_FACE:
        return None

    p_nose      = get_np_coords(landmarks[NOSE_TIP],   width, height)
    p_forehead  = get_np_coords(landmarks[FOREHEAD],   width, height)
    p_chin      = get_np_coords(landmarks[CHIN],       width, height)
    p_left_face = get_np_coords(landmarks[LEFT_FACE],  width, height)
    p_right_face= get_np_coords(landmarks[RIGHT_FACE], width, height)

    face_height = abs(p_chin[1] - p_forehead[1])

    if face_height < min_face_height:
        normalized_nose_y = None
    else:
        normalized_nose_y = float((p_nose[1] - p_forehead[1]) / face_height)

    dx = p_right_face[0] - p_left_face[0]
    dy = p_right_face[1] - p_left_face[1]
    tilt_angle = math.degrees(math.atan2(dy, dx))

    return {
        "normalized_nose_y": normalized_nose_y,
        "tilt_angle":        float(tilt_angle),
    }


# ==================================================
# 기존 시선 / 눈깜빡임 지표
# ==================================================

def get_frame_metrics(landmarks, width, height):
    l_top    = get_pixel_coords(landmarks[LEFT_EYE_TOP],    width, height)
    l_bottom = get_pixel_coords(landmarks[LEFT_EYE_BOTTOM], width, height)
    l_left   = get_pixel_coords(landmarks[LEFT_EYE_LEFT],   width, height)
    l_right  = get_pixel_coords(landmarks[LEFT_EYE_RIGHT],  width, height)

    r_top    = get_pixel_coords(landmarks[RIGHT_EYE_TOP],    width, height)
    r_bottom = get_pixel_coords(landmarks[RIGHT_EYE_BOTTOM], width, height)
    r_left   = get_pixel_coords(landmarks[RIGHT_EYE_LEFT],   width, height)
    r_right  = get_pixel_coords(landmarks[RIGHT_EYE_RIGHT],  width, height)

    nose    = get_pixel_coords(landmarks[NOSE_TIP],        width, height)
    l_pupil = get_pixel_coords(landmarks[LEFT_EYE_PUPIL],  width, height)
    r_pupil = get_pixel_coords(landmarks[RIGHT_EYE_PUPIL], width, height)

    l_vert = calculate_distance(l_top, l_bottom)
    l_horz = calculate_distance(l_left, l_right)
    r_vert = calculate_distance(r_top, r_bottom)
    r_horz = calculate_distance(r_left, r_right)

    l_ear = l_vert / l_horz if l_horz > 0 else 0
    r_ear = r_vert / r_horz if r_horz > 0 else 0
    ear   = (l_ear + r_ear) / 2.0

    dist_nose_to_left  = calculate_distance(nose, l_left)
    dist_nose_to_right = calculate_distance(nose, r_right)
    total_eye_width    = dist_nose_to_left + dist_nose_to_right

    head_turn = (
        dist_nose_to_left / total_eye_width
        if total_eye_width > 0 else 0.5
    )

    head_tilt = abs(l_left[1] - r_right[1])

    l_eye_width    = calculate_distance(l_left, l_right)
    l_pupil_offset = calculate_distance(l_left, l_pupil)

    gaze_ratio = (
        l_pupil_offset / l_eye_width
        if l_eye_width > 0 else 0.5
    )

    # 수직 시선: 눈 세로 범위 내 동공의 위치 (0=위, 1=아래, 0.5=중앙)
    # 눈이 거의 감긴 상태(모션블러·깜빡임)에서는 계산 불가능하므로 0.5로 고정
    l_vert_range = l_bottom[1] - l_top[1]
    r_vert_range = r_bottom[1] - r_top[1]
    l_gaze_y = (l_pupil[1] - l_top[1]) / l_vert_range if l_vert_range > 3 else 0.5
    r_gaze_y = (r_pupil[1] - r_top[1]) / r_vert_range if r_vert_range > 3 else 0.5
    gaze_y_ratio = (l_gaze_y + r_gaze_y) / 2.0

    return ear, head_turn, head_tilt, gaze_ratio, gaze_y_ratio


# ==================================================
# 미소 / 기존 캘리브레이션용 지표
# ==================================================

def get_smile_nod_features(landmarks, width, height):
    p_nose       = get_np_coords(landmarks[NOSE_TIP],    width, height)
    p_left_eye   = get_np_coords(landmarks[LEFT_EYE_LEFT],  width, height)
    p_right_eye  = get_np_coords(landmarks[RIGHT_EYE_RIGHT], width, height)
    p_mouth_left = get_np_coords(landmarks[MOUTH_LEFT],  width, height)
    p_mouth_right= get_np_coords(landmarks[MOUTH_RIGHT], width, height)
    p_upper_lip  = get_np_coords(landmarks[UPPER_LIP],   width, height)
    p_lower_lip  = get_np_coords(landmarks[LOWER_LIP],   width, height)

    eye_center = (p_left_eye + p_right_eye) / 2.0
    face_scale = calculate_np_distance(p_left_eye, p_right_eye)
    if face_scale <= 0:
        face_scale = 1.0

    nose_pitch_ratio = (p_nose[1] - eye_center[1]) / face_scale

    mouth_width  = calculate_np_distance(p_mouth_left, p_mouth_right)
    mouth_height = calculate_np_distance(p_upper_lip,  p_lower_lip)
    if mouth_height <= 0:
        mouth_height = 1.0

    mouth_ratio = mouth_width / mouth_height

    left_corner_raise  = (p_nose[1] - p_mouth_left[1])  / face_scale
    right_corner_raise = (p_nose[1] - p_mouth_right[1]) / face_scale
    corner_raise       = (left_corner_raise + right_corner_raise) / 2.0

    return {
        "nose_pitch_ratio": float(nose_pitch_ratio),
        "mouth_ratio":      float(mouth_ratio),
        "corner_raise":     float(corner_raise),
    }


# ==================================================
# 기존 Calibration
# ==================================================

def run_calibration(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"초기 세팅 영상을 열 수 없습니다: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ear_list       = []
    head_turn_list = []
    head_tilt_list = []
    gaze_list      = []
    gaze_y_list    = []
    pitch_ratios   = []
    mouth_ratios   = []
    corner_raises  = []

    with _make_face_landmarker() as face_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = face_landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                ear, head_turn, head_tilt, gaze_ratio, gaze_y_ratio = get_frame_metrics(
                    landmarks, width, height
                )
                ear_list.append(ear)
                head_turn_list.append(head_turn)
                head_tilt_list.append(head_tilt)
                gaze_list.append(gaze_ratio)
                gaze_y_list.append(gaze_y_ratio)

                features = get_smile_nod_features(landmarks, width, height)
                pitch_ratios.append(features["nose_pitch_ratio"])
                mouth_ratios.append(features["mouth_ratio"])
                corner_raises.append(features["corner_raise"])

    cap.release()

    if not ear_list:
        raise RuntimeError("초기 세팅 영상에서 얼굴을 감지하지 못했습니다.")

    ear_list.sort()
    start_idx      = int(len(ear_list) * 0.2)
    valid_ear_list = ear_list[start_idx:] or ear_list
    normal_ear     = sum(valid_ear_list) / len(valid_ear_list)

    return {
        "ear_threshold":       float(normal_ear * 0.75),
        "base_head_turn":      float(sum(head_turn_list) / len(head_turn_list)),
        "head_turn_tolerance": 0.07,
        "head_tilt_tolerance": float(sum(head_tilt_list) / len(head_tilt_list) + 12),
        "base_gaze_ratio":     float(sum(gaze_list) / len(gaze_list)),
        "gaze_tolerance":      0.05,
        "base_gaze_y_ratio":   float(np.mean(gaze_y_list)),
        "gaze_y_tolerance":    0.15,
        "base_pitch_ratio":    float(np.mean(pitch_ratios)),
        "base_mouth_ratio":    float(np.mean(mouth_ratios)),
        "base_corner_raise":   float(np.mean(corner_raises)),
    }


# ==================================================
# 면접 영상 내부 baseline 계산
# ==================================================

def compute_interview_baselines(
    cap,
    face_landmarker,
    pose_landmarker,
    fps,
    width,
    height,
    duration_sec,
):
    body_shoulder_baseline_sec = (
        SHORT_VIDEO_BASELINE_SEC
        if duration_sec < SHORT_VIDEO_THRESHOLD
        else BODY_SHOULDER_BASELINE_SEC
    )
    nod_baseline_sec = min(NOD_BASELINE_SEC, body_shoulder_baseline_sec)
    max_baseline_sec = max(body_shoulder_baseline_sec, nod_baseline_sec)
    max_frames       = max(1, int(fps * max_baseline_sec))

    shoulder_center_x_list = []
    shoulder_center_y_list = []
    shoulder_width_list    = []
    shoulder_angle_list    = []
    face_size_list         = []
    raw_nod_values         = []

    total_frames        = 0
    pose_detected_frames = 0
    face_detected_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened() and total_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp    = total_frames / fps if fps > 0 else 0.0
        total_frames += 1

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            face_detected_frames += 1

            if timestamp < nod_baseline_sec:
                nod_feat = extract_nod_features(face_landmarks, width, height)
                if (
                    nod_feat is not None
                    and nod_feat["normalized_nose_y"] is not None
                ):
                    raw_nod_values.append(nod_feat["normalized_nose_y"])

            if timestamp < body_shoulder_baseline_sec:
                face_size = get_face_size_from_landmarks(face_landmarks, width, height)
                if face_size is not None:
                    face_size_list.append(face_size)

        if pose_result.pose_landmarks and timestamp < body_shoulder_baseline_sec:
            pose_landmarks  = pose_result.pose_landmarks[0]
            shoulder_metrics = get_pose_shoulder_metrics(pose_landmarks, width, height)

            if shoulder_metrics is not None:
                pose_detected_frames += 1
                shoulder_center_x_list.append(shoulder_metrics["center_x"])
                shoulder_center_y_list.append(shoulder_metrics["center_y"])
                shoulder_width_list.append(shoulder_metrics["shoulder_width"])
                shoulder_angle_list.append(shoulder_metrics["shoulder_angle"])

    min_required_frames = max(10, int(fps * 3.0))

    if len(shoulder_center_x_list) < min_required_frames:
        raise RuntimeError(
            f"baseline 구간 어깨 감지 프레임 부족: "
            f"{len(shoulder_center_x_list)}프레임 감지 / 최소 {min_required_frames}프레임 필요. "
            f"영상 초반 {body_shoulder_baseline_sec:.0f}초 동안 어깨가 잘 보이는지 확인하세요."
        )

    if not raw_nod_values:
        raise RuntimeError(
            "면접 영상 초반 nod baseline 구간에서 얼굴을 감지하지 못했습니다."
        )

    rough_nod_median = float(np.median(raw_nod_values))
    nod_values = [
        v for v in raw_nod_values
        if abs(v - rough_nod_median) < CALIB_REJECT_THRESHOLD
    ]
    if not nod_values:
        nod_values = raw_nod_values

    baseline = {
        "body_shoulder_baseline_sec": float(body_shoulder_baseline_sec),
        "nod_baseline_sec":           float(nod_baseline_sec),

        "body": {
            "center_x":       float(np.median(shoulder_center_x_list)),
            "center_y":       float(np.median(shoulder_center_y_list)),
            "shoulder_width": float(np.median(shoulder_width_list)),
            "face_size":      float(np.median(face_size_list)) if face_size_list else None,
        },

        "shoulder": {
            "base_shoulder_angle":      float(np.mean(shoulder_angle_list)),
            "base_shoulder_width":      float(np.median(shoulder_width_list)),
            "shoulder_angle_tolerance": float(SHOULDER_ANGLE_TOLERANCE),
            "calibration_angle_std":    float(np.std(shoulder_angle_list)),
            "calibration_unstable":     bool(np.std(shoulder_angle_list) > 4.0),
        },

        "nod": {
            "base_nose_y":           float(np.mean(nod_values)),
            "calibration_pitch_std": float(np.std(nod_values)),
            "calibration_unstable":  bool(np.std(nod_values) > 0.025),
            "used_frames":           int(len(nod_values)),
            "raw_frames":            int(len(raw_nod_values)),
        },

        "quality": {
            "baseline_total_frames":  int(total_frames),
            "pose_detected_frames":   int(pose_detected_frames),
            "face_detected_frames":   int(face_detected_frames),
            "pose_detection_ratio":   round(pose_detected_frames / max(total_frames, 1) * 100, 2),
            "face_detection_ratio":   round(face_detected_frames / max(total_frames, 1) * 100, 2),
        },
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return baseline


# ==================================================
# Analysis
# ==================================================

def analyze_behavior_video(video_path, config):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"면접 영상을 열 수 없습니다: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1:
        fps = 30.0

    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_sec_by_meta = (
        frame_count / fps
        if fps > 0 and frame_count > 0 else 0.0
    )

    frame_details = []

    blink_count       = 0
    is_eye_closed_prev = False

    sample_interval = max(1, int(fps / 5))
    frame_idx       = 0

    # ==================================================
    # Smile
    # ==================================================
    smile_score_accumulator    = 0.0
    analyzed_face_frame_count  = 0
    score_buffer               = deque(maxlen=max(1, int(fps * 0.3)))

    # ==================================================
    # Nod 상태 변수
    # ==================================================
    nod_count            = 0
    nod_state            = "IDLE"
    local_peak_delta     = 0.0
    nod_smooth_buffer    = deque(maxlen=NOD_SMOOTHING_WINDOW)
    nod_velocity_buffer  = deque(maxlen=VELOCITY_WINDOW)
    prev_smooth_nose_y   = None
    last_count_video_sec = -999.0

    # ==================================================
    # [수정] Body sway 상태 변수
    # lr_sway_count / fb_sway_count 분리 제거.
    # body_sway_count 단일 카운터 + 쿨다운으로 중복 방지.
    # ==================================================
    lr_ratio_buffer       = []
    fb_width_ratio_buffer = []
    fb_y_ratio_buffer     = []
    fb_face_ratio_buffer  = []

    lr_state = "NEUTRAL"   # NEUTRAL / LEFT / RIGHT
    fb_state = "NEUTRAL"   # NEUTRAL / FORWARD / BACKWARD

    body_sway_count          = 0           # [수정] 단일 통합 카운터
    last_sway_timestamp      = -999.0      # [수정] 쿨다운 추적용

    body_direction_counts = {
        "LR_LEFT":    0,
        "LR_RIGHT":   0,
        "FB_FORWARD":  0,
        "FB_BACKWARD": 0,
    }

    # ==================================================
    # Shoulder 상태 변수
    # ==================================================
    shoulder_angle_history    = deque(maxlen=SHOULDER_SMOOTHING_FRAMES)
    shoulder_stable_frames    = 0
    shoulder_tilted_frames    = 0
    shoulder_detected_frames  = 0
    shoulder_not_detected_frames = 0

    tilt_count               = 0
    tilted_candidate_frames  = 0
    stable_candidate_frames  = 0
    current_state_tilted     = False

    # ==================================================
    # config 검증
    # ==================================================
    required_keys = [
        "ear_threshold",
        "base_head_turn",
        "head_turn_tolerance",
        "head_tilt_tolerance",
        "base_gaze_ratio",
        "gaze_tolerance",
        "base_mouth_ratio",
        "base_corner_raise",
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise RuntimeError(f"캘리브레이션 설정값이 누락되었습니다: {missing_keys}")

    with _make_face_landmarker() as face_landmarker, \
         _make_pose_landmarker() as pose_landmarker:

        interview_baseline = compute_interview_baselines(
            cap=cap,
            face_landmarker=face_landmarker,
            pose_landmarker=pose_landmarker,
            fps=fps,
            width=width,
            height=height,
            duration_sec=duration_sec_by_meta,
        )

        body_baseline    = interview_baseline["body"]
        shoulder_baseline = interview_baseline["shoulder"]
        nod_baseline     = interview_baseline["nod"]

        body_shoulder_baseline_sec = interview_baseline["body_shoulder_baseline_sec"]
        nod_baseline_sec           = interview_baseline["nod_baseline_sec"]

        base_sw      = body_baseline["shoulder_width"]
        base_fs      = body_baseline["face_size"]
        base_nose_y  = nod_baseline["base_nose_y"]
        base_shoulder_angle = shoulder_baseline["base_shoulder_angle"]

        if base_sw < 1:
            raise RuntimeError("면접 영상 초반 baseline의 어깨 너비가 너무 작습니다.")

        print("=" * 50)
        print("interview baseline")
        print("video:", video_path)
        print("fps:", fps)
        print("duration_sec(meta):", duration_sec_by_meta)
        print("body/shoulder baseline sec:", body_shoulder_baseline_sec)
        print("nod baseline sec:", nod_baseline_sec)
        print("body baseline:", body_baseline)
        print("shoulder baseline:", shoulder_baseline)
        print("nod baseline:", nod_baseline)
        print("=" * 50)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            is_body_sway_event    = False
            body_event            = "NONE"
            is_shoulder_detected  = False
            current_shoulder_angle = 0.0
            shoulder_angle_diff   = 0.0
            shoulder_width_value  = 0.0
            shoulder_tilt_value   = 0.0
            is_shoulder_stable    = True

            detail = {
                "timestamp": round(timestamp, 2),

                "is_blink":    False,
                "ear_value":   0.0,

                "gaze_direction": "center",
                "head_turn":  0.0,
                "head_tilt":  0.0,
                "gaze_ratio": 0.0,

                "shoulder_tilt":         0.0,
                "shoulder_width":        0.0,
                "is_swaying":            False,
                "shoulder_stable":       True,
                "shoulder_stable_frame": True,

                "smile_ratio": 0.0,
                "is_smiling":  False,

                "is_nodding": False,
                "nod_delta":  0.0,

                # 디버깅/검증용
                "body_event":         "NONE",
                "body_sway_count":    body_sway_count,
                "shoulder_angle":     0.0,
                "shoulder_angle_diff": 0.0,
            }

            face_landmarks = None
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]

                ear, head_turn, head_tilt, gaze_ratio, gaze_y_ratio = get_frame_metrics(
                    face_landmarks, width, height
                )

                detail["ear_value"]    = float(ear)
                detail["head_turn"]    = float(head_turn)
                detail["head_tilt"]    = float(head_tilt)
                detail["gaze_ratio"]   = float(gaze_ratio)
                detail["gaze_y_ratio"] = float(gaze_y_ratio)

                # ==================================================
                # Blink
                # ==================================================
                is_closed = ear < config["ear_threshold"]
                if is_eye_closed_prev and not is_closed:
                    blink_count   += 1
                    detail["is_blink"] = True
                is_eye_closed_prev = is_closed

                # ==================================================
                # Gaze
                # ==================================================
                is_head_turn_ok  = abs(head_turn - config["base_head_turn"]) < config["head_turn_tolerance"]
                is_head_tilt_ok  = head_tilt < config["head_tilt_tolerance"]
                is_iris_front    = abs(gaze_ratio - config["base_gaze_ratio"]) < config["gaze_tolerance"]
                is_iris_y_front  = abs(gaze_y_ratio - config.get("base_gaze_y_ratio", 0.5)) < config.get("gaze_y_tolerance", 0.15)

                detail["gaze_direction"] = (
                    "center"
                    if (is_head_turn_ok and is_head_tilt_ok and is_iris_front and is_iris_y_front)
                    else "deviated"
                )

                # ==================================================
                # Smile
                # ==================================================
                features = get_smile_nod_features(face_landmarks, width, height)
                analyzed_face_frame_count += 1

                ratio_delta  = features["mouth_ratio"]  - config["base_mouth_ratio"]
                corner_delta = features["corner_raise"] - config["base_corner_raise"]
                smile_score  = ratio_delta * 0.1 + corner_delta * 0.9

                score_buffer.append(smile_score)
                smooth_score = float(np.mean(score_buffer))
                detail["smile_ratio"] = float(max(0.0, smooth_score))

                if corner_delta > 0.025 and smooth_score > 0.018:
                    detail["is_smiling"] = True
                    smile_score_accumulator += 1.0
                elif corner_delta > 0.008 and smooth_score > 0.005:
                    detail["is_smiling"] = True
                    smile_score_accumulator += 0.7
                else:
                    detail["is_smiling"] = False

                # ==================================================
                # Nod
                # ==================================================
                nod_feat = extract_nod_features(face_landmarks, width, height)

                if timestamp < nod_baseline_sec:
                    if (
                        nod_feat is not None
                        and nod_feat["normalized_nose_y"] is not None
                    ):
                        nod_smooth_buffer.append(nod_feat["normalized_nose_y"])
                    prev_smooth_nose_y = None

                elif (
                    nod_feat is not None
                    and nod_feat["normalized_nose_y"] is not None
                ):
                    normalized_nose_y = nod_feat["normalized_nose_y"]
                    tilt_angle        = nod_feat["tilt_angle"]

                    nod_smooth_buffer.append(normalized_nose_y)
                    smooth_nose_y = sum(nod_smooth_buffer) / len(nod_smooth_buffer)

                    nod_delta = smooth_nose_y - base_nose_y
                    detail["nod_delta"] = float(nod_delta)

                    avg_velocity = 0.0
                    if prev_smooth_nose_y is not None:
                        velocity = smooth_nose_y - prev_smooth_nose_y
                        nod_velocity_buffer.append(velocity)
                        if nod_velocity_buffer:
                            avg_velocity = (
                                sum(nod_velocity_buffer) / len(nod_velocity_buffer)
                            )

                    if nod_state == "IDLE":
                        if USE_IGNORE_UP_FILTER and nod_delta < UP_IGNORE_THRESHOLD:
                            nod_state = "IGNORE_UP"

                        elif (
                            nod_delta > DOWN_THRESHOLD
                            and avg_velocity > VELOCITY_DOWN_THRESHOLD
                        ):
                            nod_state        = "DESCENDING"
                            local_peak_delta = nod_delta

                    elif nod_state == "DESCENDING":
                        local_peak_delta   = max(local_peak_delta, nod_delta)
                        direction_reversed = avg_velocity < VELOCITY_UP_THRESHOLD

                        if direction_reversed:
                            elapsed_since_last = timestamp - last_count_video_sec

                            if (
                                local_peak_delta >= MIN_NOD_DEPTH
                                and elapsed_since_last >= COOLDOWN_SEC
                            ):
                                nod_count            += 1
                                last_count_video_sec  = timestamp
                                nod_state             = "ASCENDING"
                                detail["is_nodding"]  = True
                                local_peak_delta      = 0.0
                            else:
                                nod_state        = "IDLE"
                                local_peak_delta = 0.0

                    elif nod_state == "ASCENDING":
                        if (
                            nod_delta > DOWN_THRESHOLD
                            and avg_velocity > VELOCITY_DOWN_THRESHOLD
                        ):
                            nod_state        = "DESCENDING"
                            local_peak_delta = nod_delta

                        elif nod_delta < DOWN_THRESHOLD * IDLE_RETURN_RATIO:
                            nod_state = "IDLE"

                    elif nod_state == "IGNORE_UP":
                        if abs(nod_delta) < BASELINE_RETURN_THRESHOLD:
                            nod_state = "IDLE"

                    prev_smooth_nose_y = smooth_nose_y

                else:
                    prev_smooth_nose_y = None

            else:
                if timestamp >= nod_baseline_sec:
                    prev_smooth_nose_y = None

            # ==================================================
            # Pose 기반 몸통 흔들림 / 어깨 안정성
            # ==================================================
            pose_landmarks   = None
            shoulder_metrics = None

            if pose_result.pose_landmarks:
                pose_landmarks   = pose_result.pose_landmarks[0]
                shoulder_metrics = get_pose_shoulder_metrics(pose_landmarks, width, height)

            if shoulder_metrics is not None:
                is_shoulder_detected  = True
                shoulder_center_x     = shoulder_metrics["center_x"]
                shoulder_center_y     = shoulder_metrics["center_y"]
                shoulder_width_value  = shoulder_metrics["shoulder_width"]
                current_shoulder_angle = shoulder_metrics["shoulder_angle"]

                l_sh = pose_landmarks[LEFT_SHOULDER]
                r_sh = pose_landmarks[RIGHT_SHOULDER]
                shoulder_tilt_value = float(l_sh.y - r_sh.y)

                detail["shoulder_tilt"]  = float(shoulder_tilt_value)
                detail["shoulder_width"] = float(shoulder_width_value)

                # --------------------------------------------------
                # Body sway — baseline 구간은 카운트 제외
                # --------------------------------------------------
                if timestamp >= body_shoulder_baseline_sec:
                    current_lr_ratio = (
                        shoulder_center_x - body_baseline["center_x"]
                    ) / base_sw

                    current_fb_width_ratio = (
                        shoulder_width_value - base_sw
                    ) / base_sw

                    current_fb_y_ratio = (
                        shoulder_center_y - body_baseline["center_y"]
                    ) / base_sw

                    face_size = (
                        get_face_size_from_landmarks(face_landmarks, width, height)
                        if face_landmarks is not None else None
                    )
                    if face_size is not None and base_fs is not None and base_fs > 1:
                        current_fb_face_ratio = (face_size - base_fs) / base_fs
                    else:
                        current_fb_face_ratio = 0.0

                    lr_ratio_buffer       = update_buffer(lr_ratio_buffer,       current_lr_ratio,       BODY_SMOOTHING_FRAMES)
                    fb_width_ratio_buffer = update_buffer(fb_width_ratio_buffer, current_fb_width_ratio, BODY_SMOOTHING_FRAMES)
                    fb_y_ratio_buffer     = update_buffer(fb_y_ratio_buffer,     current_fb_y_ratio,     BODY_SMOOTHING_FRAMES)
                    fb_face_ratio_buffer  = update_buffer(fb_face_ratio_buffer,  current_fb_face_ratio,  BODY_SMOOTHING_FRAMES)

                    smooth_lr_ratio       = get_median(lr_ratio_buffer)
                    smooth_fb_width_ratio = get_median(fb_width_ratio_buffer)
                    smooth_fb_y_ratio     = get_median(fb_y_ratio_buffer)
                    smooth_fb_face_ratio  = get_median(fb_face_ratio_buffer)

                    fb_score = calculate_fb_score(
                        smooth_fb_width_ratio,
                        smooth_fb_y_ratio,
                        smooth_fb_face_ratio,
                    )

                    lr_is_active = abs(smooth_lr_ratio) >= LR_SWAY_THRESHOLD * LR_ACTIVE_RATIO

                    # --------------------------------------------------
                    # [수정] 좌우 상태 머신 — 이벤트 감지만 담당
                    # 실제 카운트는 아래 통합 쿨다운 블록에서 처리
                    # --------------------------------------------------
                    lr_triggered = False

                    if lr_state == "NEUTRAL":
                        if smooth_lr_ratio >= LR_SWAY_THRESHOLD:
                            lr_state = "RIGHT"
                            lr_triggered = True
                            body_direction_counts["LR_RIGHT"] += 1
                            body_event = "LR_RIGHT_SWAY"

                        elif smooth_lr_ratio <= -LR_SWAY_THRESHOLD:
                            lr_state = "LEFT"
                            lr_triggered = True
                            body_direction_counts["LR_LEFT"] += 1
                            body_event = "LR_LEFT_SWAY"

                    elif lr_state == "RIGHT":
                        if abs(smooth_lr_ratio) <= LR_RETURN_THRESHOLD:
                            lr_state = "NEUTRAL"

                    elif lr_state == "LEFT":
                        if abs(smooth_lr_ratio) <= LR_RETURN_THRESHOLD:
                            lr_state = "NEUTRAL"

                    # --------------------------------------------------
                    # [수정] 앞뒤 상태 머신 — 이벤트 감지만 담당
                    # --------------------------------------------------
                    fb_triggered = False

                    if fb_state == "NEUTRAL":
                        if not lr_is_active:
                            if fb_score >= FB_SCORE_THRESHOLD:
                                fb_state = "FORWARD"
                                fb_triggered = True
                                body_direction_counts["FB_FORWARD"] += 1
                                body_event = (
                                    "FB_FORWARD" if body_event == "NONE"
                                    else body_event + "+FB_FORWARD"
                                )

                            elif fb_score <= -FB_SCORE_THRESHOLD:
                                fb_state = "BACKWARD"
                                fb_triggered = True
                                body_direction_counts["FB_BACKWARD"] += 1
                                body_event = (
                                    "FB_BACKWARD" if body_event == "NONE"
                                    else body_event + "+FB_BACKWARD"
                                )

                    elif fb_state in ["FORWARD", "BACKWARD"]:
                        if abs(fb_score) <= FB_SCORE_RETURN_THRESHOLD:
                            fb_state = "NEUTRAL"

                    # --------------------------------------------------
                    # [수정] 통합 쿨다운 카운터
                    # LR/FB 중 하나라도 새 이벤트가 감지되면,
                    # 쿨다운 이후일 때만 body_sway_count 1 증가.
                    # 두 방향 동시 감지 → 여전히 1번만 카운트.
                    # --------------------------------------------------
                    if lr_triggered or fb_triggered:
                        elapsed_since_sway = timestamp - last_sway_timestamp
                        if elapsed_since_sway >= BODY_SWAY_COOLDOWN_SEC:
                            body_sway_count     += 1
                            last_sway_timestamp  = timestamp
                            is_body_sway_event   = True

                # --------------------------------------------------
                # Shoulder stability — baseline 구간 제외
                # --------------------------------------------------
                if timestamp >= body_shoulder_baseline_sec:
                    shoulder_detected_frames += 1

                    shoulder_angle_history.append(current_shoulder_angle)
                    smooth_shoulder_angle = (
                        sum(shoulder_angle_history) / len(shoulder_angle_history)
                    )
                    shoulder_angle_diff = abs(smooth_shoulder_angle - base_shoulder_angle)
                    angle_unstable = shoulder_angle_diff > SHOULDER_ANGLE_TOLERANCE

                    # 위치 이탈: body sway 블록에서 이미 업데이트된 스무딩 값 재사용
                    pos_unstable = (
                        abs(smooth_lr_ratio) > SHOULDER_POSITION_TOLERANCE
                        or abs(smooth_fb_y_ratio) > SHOULDER_POSITION_TOLERANCE
                    )

                    raw_is_tilted = angle_unstable or pos_unstable

                    if raw_is_tilted:
                        tilted_candidate_frames += 1
                        stable_candidate_frames  = 0
                    else:
                        stable_candidate_frames += 1
                        tilted_candidate_frames  = 0

                    if not current_state_tilted and tilted_candidate_frames >= REQUIRED_TILT_FRAMES:
                        current_state_tilted = True
                        tilt_count += 1

                    elif current_state_tilted and stable_candidate_frames >= REQUIRED_STABLE_FRAMES:
                        current_state_tilted = False

                    is_shoulder_stable = not current_state_tilted

                    if is_shoulder_stable:
                        shoulder_stable_frames += 1
                    else:
                        shoulder_tilted_frames += 1
                else:
                    is_shoulder_stable = True

                detail["shoulder_stable"]       = bool(is_shoulder_stable)
                detail["shoulder_stable_frame"] = bool(is_shoulder_stable)
                detail["shoulder_angle"]        = float(current_shoulder_angle)
                detail["shoulder_angle_diff"]   = float(shoulder_angle_diff)

            else:
                if timestamp >= body_shoulder_baseline_sec:
                    shoulder_not_detected_frames += 1

                detail["shoulder_stable"]       = False
                detail["shoulder_stable_frame"] = False

            detail["is_swaying"]      = bool(is_body_sway_event)
            detail["body_event"]      = body_event
            detail["body_sway_count"] = int(body_sway_count)

            if frame_idx % sample_interval == 0:
                frame_details.append(detail)

            frame_idx += 1

    cap.release()

    total_sampled_frames = len(frame_details)

    duration_sec = (
        frame_idx / fps if fps > 0 else duration_sec_by_meta
    )
    duration_min = duration_sec / 60 if duration_sec > 0 else 1.0

    focus_rate = (
        sum(1 for d in frame_details if d["gaze_direction"] == "center")
        / total_sampled_frames * 100
        if total_sampled_frames > 0 else 0.0
    )
    deviated_rate = 100.0 - focus_rate

    total_smile_rate = (
        smile_score_accumulator / analyzed_face_frame_count * 100
        if analyzed_face_frame_count > 0 else 0.0
    )

    shoulder_stability_ratio = (
        shoulder_stable_frames / shoulder_detected_frames * 100
        if shoulder_detected_frames > 0 else 0.0
    )

    blinks_per_min    = blink_count      / duration_min if duration_min > 0 else 0.0
    body_sway_per_min = body_sway_count  / duration_min if duration_min > 0 else 0.0
    nod_per_min       = nod_count        / duration_min if duration_min > 0 else 0.0

    print("=" * 50)
    print("video:", video_path)
    print("fps:", fps)
    print("frame_idx:", frame_idx)
    print("duration_sec:", duration_sec)
    print("body_sway_count:", body_sway_count)
    print("body_sway_per_min(full duration):", body_sway_per_min)
    print("shoulder_stability_ratio:", shoulder_stability_ratio)
    print("nod_count:", nod_count)
    print("=" * 50)

    summary = {
        "focus_rate":          round(focus_rate, 1),
        "deviated_gaze_rate":  round(deviated_rate, 1),

        "blink_count":         int(blink_count),
        "blinks_per_min":      round(blinks_per_min, 1),

        "nod_count":           int(nod_count),
        "nod_per_min":         round(nod_per_min, 1),

        "shoulder_stability":       round(shoulder_stability_ratio, 1),
        "shoulder_stability_ratio": round(shoulder_stability_ratio, 1),

        "shoulder_tilt_count":          int(tilt_count),
        "shoulder_stable_frames":       int(shoulder_stable_frames),
        "shoulder_tilted_frames":       int(shoulder_tilted_frames),
        "shoulder_detected_frames":     int(shoulder_detected_frames),
        "shoulder_not_detected_frames": int(shoulder_not_detected_frames),

        # [수정] lr_sway_count / fb_sway_count 제거, body_sway_count 단일 필드만 유지
        "body_sway_count":   int(body_sway_count),
        "body_sway_per_min": round(body_sway_per_min, 1),

        "total_smile_rate": round(total_smile_rate, 1),
        "duration_sec":     round(duration_sec, 2),

        "interview_baseline":         interview_baseline,
        "body_sway_direction_counts": body_direction_counts,
    }

    return frame_details, summary