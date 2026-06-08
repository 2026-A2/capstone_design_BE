import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import subprocess
import json
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
        "head_turn_tolerance": 0.10,
        "head_tilt_tolerance": float(sum(head_tilt_list) / len(head_tilt_list) + 15),
        "base_gaze_ratio":     float(sum(gaze_list) / len(gaze_list)),
        "gaze_tolerance":      0.08,
        "base_gaze_y_ratio":   float(np.mean(gaze_y_list)),
        "gaze_y_tolerance":    0.20,
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
    # duration_sec가 None이면 메타데이터를 못 읽은 것(WebM 등) → 긴 영상으로 간주
    body_shoulder_baseline_sec = (
        SHORT_VIDEO_BASELINE_SEC
        if duration_sec is not None and duration_sec < SHORT_VIDEO_THRESHOLD
        else BODY_SHOULDER_BASELINE_SEC
    )
    nod_baseline_sec = min(NOD_BASELINE_SEC, body_shoulder_baseline_sec)

    # 최소 요구 프레임: 0.5초분이면 안정적인 중앙값 계산 가능
    MIN_SHOULDER_FRAMES = max(5, int(fps * 0.5))
    MIN_NOD_FRAMES      = 3
    # 초반 탐지 실패 시 최대 30초까지 탐색 확장
    MAX_SEARCH_FRAMES   = int(fps * 30.0)

    shoulder_center_x_list = []
    shoulder_center_y_list = []
    shoulder_width_list    = []
    shoulder_angle_list    = []
    face_size_list         = []
    raw_nod_values         = []
    raw_corner_raises      = []
    raw_mouth_ratios       = []

    total_frames         = 0
    pose_detected_frames = 0
    face_detected_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened() and total_frames < MAX_SEARCH_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            face_detected_frames += 1

            # 필요한 만큼만 수집
            if len(raw_nod_values) < int(fps * nod_baseline_sec) + 10:
                nod_feat = extract_nod_features(face_landmarks, width, height)
                if (
                    nod_feat is not None
                    and nod_feat["normalized_nose_y"] is not None
                ):
                    raw_nod_values.append(nod_feat["normalized_nose_y"])

            if len(face_size_list) < int(fps * body_shoulder_baseline_sec) + 10:
                face_size = get_face_size_from_landmarks(face_landmarks, width, height)
                if face_size is not None:
                    face_size_list.append(face_size)

            if len(raw_corner_raises) < int(fps * nod_baseline_sec) + 10:
                smile_feat = get_smile_nod_features(face_landmarks, width, height)
                raw_corner_raises.append(smile_feat["corner_raise"])
                raw_mouth_ratios.append(smile_feat["mouth_ratio"])

        if (
            pose_result.pose_landmarks
            and len(shoulder_center_x_list) < int(fps * body_shoulder_baseline_sec) + 10
        ):
            pose_landmarks   = pose_result.pose_landmarks[0]
            shoulder_metrics = get_pose_shoulder_metrics(pose_landmarks, width, height)

            if shoulder_metrics is not None:
                pose_detected_frames += 1
                shoulder_center_x_list.append(shoulder_metrics["center_x"])
                shoulder_center_y_list.append(shoulder_metrics["center_y"])
                shoulder_width_list.append(shoulder_metrics["shoulder_width"])
                shoulder_angle_list.append(shoulder_metrics["shoulder_angle"])

        # 충분한 baseline 확보 시 조기 종료
        if (
            len(shoulder_center_x_list) >= MIN_SHOULDER_FRAMES
            and len(raw_nod_values) >= MIN_NOD_FRAMES
        ):
            break

    # body/shoulder baseline — 어깨 감지 불충분 시 None
    body_baseline     = None
    shoulder_baseline = None
    if len(shoulder_center_x_list) >= MIN_SHOULDER_FRAMES:
        body_baseline = {
            "center_x":       float(np.median(shoulder_center_x_list)),
            "center_y":       float(np.median(shoulder_center_y_list)),
            "shoulder_width": float(np.median(shoulder_width_list)),
            "face_size":      float(np.median(face_size_list)) if face_size_list else None,
        }
        shoulder_baseline = {
            "base_shoulder_angle":      float(np.mean(shoulder_angle_list)),
            "base_shoulder_width":      float(np.median(shoulder_width_list)),
            "shoulder_angle_tolerance": float(SHOULDER_ANGLE_TOLERANCE),
            "calibration_angle_std":    float(np.std(shoulder_angle_list)),
            "calibration_unstable":     bool(np.std(shoulder_angle_list) > 4.0),
        }

    # nod baseline — 얼굴 감지 불충분 시 None
    nod_baseline = None
    if len(raw_nod_values) >= MIN_NOD_FRAMES:
        rough_nod_median = float(np.median(raw_nod_values))
        nod_values = [
            v for v in raw_nod_values
            if abs(v - rough_nod_median) < CALIB_REJECT_THRESHOLD
        ]
        if not nod_values:
            nod_values = raw_nod_values
        nod_baseline = {
            "base_nose_y":           float(np.mean(nod_values)),
            "calibration_pitch_std": float(np.std(nod_values)),
            "calibration_unstable":  bool(np.std(nod_values) > 0.025),
            "used_frames":           int(len(nod_values)),
            "raw_frames":            int(len(raw_nod_values)),
        }

    # smile baseline — 면접 영상 초반 3초 기준으로 고개 위치 보정
    # 캘리브레이션(중앙)과 면접(우측) 화면 배치 차이로 인한 고개 각도 오프셋 흡수
    smile_baseline = None
    if len(raw_corner_raises) >= MIN_NOD_FRAMES:
        smile_baseline = {
            "base_corner_raise": float(np.median(raw_corner_raises)),
            "base_mouth_ratio":  float(np.median(raw_mouth_ratios)),
        }

    return {
        "body_shoulder_baseline_sec": float(body_shoulder_baseline_sec),
        "nod_baseline_sec":           float(nod_baseline_sec),
        "body":     body_baseline,
        "shoulder": shoulder_baseline,
        "nod":      nod_baseline,
        "smile":    smile_baseline,
        "quality": {
            "baseline_total_frames":  int(total_frames),
            "pose_detected_frames":   int(pose_detected_frames),
            "face_detected_frames":   int(face_detected_frames),
            "pose_detection_ratio":   round(pose_detected_frames / max(total_frames, 1) * 100, 2),
            "face_detection_ratio":   round(face_detected_frames / max(total_frames, 1) * 100, 2),
        },
    }


# ==================================================
# ffprobe 기반 영상 정보 조회 (WebM 등 컨테이너 메타 오류 대응)
# ==================================================

def _get_video_info_ffprobe(video_path: str):
    """
    ffprobe로 실제 FPS와 영상 길이를 읽는다.
    Chrome WebM은 r_frame_rate / CAP_PROP_FPS 가 TrackEntry DefaultDuration 기반으로
    실제 값의 2배로 잘못 기록되는 버그가 있어, 실제 패킷 PTS 타임스탬프에서 FPS를 계산.
    실패 시 (None, None) 반환.
    """
    fps = None
    duration = None

    try:
        # 1. 컨테이너(format) 레벨 duration 조회
        #    Chrome WebM은 Segment Info에 올바른 duration을 기록함
        fmt_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, timeout=10,
        )
        fmt_data = json.loads(fmt_result.stdout)
        duration = float(fmt_data.get("format", {}).get("duration") or 0) or None
    except Exception:
        pass

    try:
        # 2. 실제 패킷 PTS 타임스탬프로 FPS 계산
        #    r_frame_rate는 DefaultDuration 기반(부정확)이므로 사용하지 않음
        #    실제 Block timestamp를 읽어 프레임 간격의 중앙값으로 FPS 계산
        pts_result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "packet=pts_time",
                "-read_intervals", "%+5",   # 처음 5초 분량 패킷만 읽음
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        timestamps = [
            float(t) for t in pts_result.stdout.strip().split("\n")
            if t.strip() and t.strip() != "N/A"
        ]
        if len(timestamps) >= 10:
            intervals = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
                if timestamps[i + 1] > timestamps[i]
            ]
            if intervals:
                median_interval = float(np.median(intervals))
                if median_interval > 0:
                    measured = 1.0 / median_interval
                    if 1.0 < measured < 200.0:
                        fps = measured
    except Exception:
        pass

    return fps, duration


# ==================================================
# FPS 측정
# ==================================================

def _measure_actual_fps(cap, sample_frames: int = 60):
    """
    POS_MSEC으로 실제 프레임 간격을 측정해 FPS 반환.
    측정 불가 시 None 반환. 호출 후 cap은 소비된 상태 → 재오픈 필요.
    """
    timestamps_ms = []
    for _ in range(sample_frames + 1):
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, _ = cap.read()
        if not ret:
            break
        if ts >= 0:
            timestamps_ms.append(ts)

    if len(timestamps_ms) < 10:
        return None

    intervals = [
        timestamps_ms[i + 1] - timestamps_ms[i]
        for i in range(len(timestamps_ms) - 1)
        if timestamps_ms[i + 1] > timestamps_ms[i]
    ]
    if not intervals:
        return None

    median_ms = float(np.median(intervals))
    if median_ms <= 0:
        return None

    measured = 1000.0 / median_ms
    return measured if 1.0 < measured < 200.0 else None


# ==================================================
# Analysis
# ==================================================

def analyze_behavior_video(video_path, config):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"면접 영상을 열 수 없습니다: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)

    # ffprobe로 실제 FPS와 영상 길이 우선 조회
    # (Chrome WebM은 OpenCV가 FPS를 2배 오독하는 버그가 있어 ffprobe를 우선 신뢰)
    ffprobe_fps, ffprobe_duration = _get_video_info_ffprobe(video_path)

    if ffprobe_fps:
        fps = ffprobe_fps
    elif raw_fps and 1.0 < raw_fps <= 120.0:
        fps = raw_fps
    else:
        measured = _measure_actual_fps(cap)
        cap.release()
        cap = cv2.VideoCapture(video_path)
        fps = measured if measured else 30.0

    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ffprobe duration 우선 사용; 없으면 메타데이터 frame_count/fps, 그것도 없으면 None
    duration_sec_by_meta = (
        ffprobe_duration
        or (frame_count / fps if fps > 0 and frame_count > 0 else None)
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

        body_baseline     = interview_baseline["body"]
        shoulder_baseline = interview_baseline["shoulder"]
        nod_baseline      = interview_baseline["nod"]
        smile_baseline    = interview_baseline["smile"]

        body_shoulder_baseline_sec = interview_baseline["body_shoulder_baseline_sec"]
        nod_baseline_sec           = interview_baseline["nod_baseline_sec"]

        # baseline 가용 여부 — None이면 해당 지표는 0으로 처리
        can_analyze_body = body_baseline is not None
        can_analyze_nod  = nod_baseline  is not None

        if can_analyze_body:
            base_sw             = body_baseline["shoulder_width"]
            base_fs             = body_baseline["face_size"]
            base_shoulder_angle = shoulder_baseline["base_shoulder_angle"]
            if base_sw < 1:
                can_analyze_body = False
        else:
            base_sw = base_fs = base_shoulder_angle = None

        base_nose_y = nod_baseline["base_nose_y"] if can_analyze_nod else None

        # smile 기준: 면접 영상 초반 baseline 우선, 없으면 캘리브레이션 fallback
        base_corner_raise = (
            smile_baseline["base_corner_raise"]
            if smile_baseline else config.get("base_corner_raise", 0.0)
        )
        base_mouth_ratio = (
            smile_baseline["base_mouth_ratio"]
            if smile_baseline else config.get("base_mouth_ratio", 1.0)
        )

        print("=" * 50)
        print("interview baseline")
        print("video:", video_path)
        print("fps:", fps, f"(raw: {raw_fps}, ffprobe: {ffprobe_fps})")
        print("duration_sec(ffprobe):", ffprobe_duration, "/ (meta):", duration_sec_by_meta)
        print("body/shoulder baseline sec:", body_shoulder_baseline_sec)
        print("nod baseline sec:", nod_baseline_sec)
        print("body baseline:", body_baseline)
        print("shoulder baseline:", shoulder_baseline)
        print("nod baseline:", nod_baseline)
        print("=" * 50)

        # webm(VP9) 파일은 seek가 동작하지 않아 재오픈
        cap.release()
        cap = cv2.VideoCapture(video_path)

        last_frame_ts_ms = 0.0

        while cap.isOpened():
            frame_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_ts_ms > 0:
                last_frame_ts_ms = frame_ts_ms

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

                ratio_delta  = features["mouth_ratio"]  - base_mouth_ratio
                corner_delta = features["corner_raise"] - base_corner_raise
                smile_score  = ratio_delta * 0.1 + corner_delta * 0.9

                score_buffer.append(smile_score)
                smooth_score = float(np.mean(score_buffer))
                detail["smile_ratio"] = float(max(0.0, smooth_score))

                if corner_delta > 0.030 and smooth_score > 0.022:
                    detail["is_smiling"] = True
                    smile_score_accumulator += 1.0
                elif corner_delta > 0.015 and smooth_score > 0.010:
                    detail["is_smiling"] = True
                    smile_score_accumulator += 0.7
                else:
                    detail["is_smiling"] = False

                # ==================================================
                # Nod
                # ==================================================
                nod_feat = (
                    extract_nod_features(face_landmarks, width, height)
                    if can_analyze_nod else None
                )

                if can_analyze_nod and timestamp < nod_baseline_sec:
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
                if can_analyze_nod and timestamp >= nod_baseline_sec:
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
                # Body sway — baseline 없으면 스킵
                # --------------------------------------------------
                smooth_lr_ratio   = 0.0
                smooth_fb_y_ratio = 0.0

                if can_analyze_body and timestamp >= body_shoulder_baseline_sec:
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

                    if lr_triggered or fb_triggered:
                        elapsed_since_sway = timestamp - last_sway_timestamp
                        if elapsed_since_sway >= BODY_SWAY_COOLDOWN_SEC:
                            body_sway_count     += 1
                            last_sway_timestamp  = timestamp
                            is_body_sway_event   = True

                # --------------------------------------------------
                # Shoulder stability — baseline 없으면 스킵
                # --------------------------------------------------
                if can_analyze_body and timestamp >= body_shoulder_baseline_sec:
                    shoulder_detected_frames += 1

                    shoulder_angle_history.append(current_shoulder_angle)
                    smooth_shoulder_angle = (
                        sum(shoulder_angle_history) / len(shoulder_angle_history)
                    )
                    shoulder_angle_diff = abs(smooth_shoulder_angle - base_shoulder_angle)
                    angle_unstable = shoulder_angle_diff > SHOULDER_ANGLE_TOLERANCE

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

    # ffprobe duration 우선 사용 (가장 정확)
    # 없으면 frame_idx / fps로 추정
    if ffprobe_duration:
        duration_sec = ffprobe_duration
    elif fps > 0:
        duration_sec = frame_idx / fps
    else:
        duration_sec = 0.0
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
    print("focus_rate:", round(focus_rate, 1), "%")
    print("blink_count:", blink_count, f"({round(blinks_per_min, 1)}/min)")
    print("total_smile_rate:", round(total_smile_rate, 1), "%")
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