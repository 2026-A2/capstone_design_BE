"""
행동 분석 독립 실행 스크립트
서버 없이 초기 세팅 영상과 면접 영상만으로 분석 결과를 확인합니다.

사용법:
    python behavior_demo.py
    python behavior_demo.py --calib media/calib.mp4 --interview media/interview.mp4
    python behavior_demo.py --calib media/calib.mp4 --interview media/q1.mp4 media/q2.mp4
"""

import argparse
import json
import os
import sys

import numpy as np


MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")

DEFAULT_CALIB_NAMES     = ["calib.mp4", "calib.webm", "init.mp4", "init.webm"]
DEFAULT_INTERVIEW_NAMES = ["interview.mp4", "interview.webm"]

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def find_default_video(candidates: list[str]) -> str | None:
    for name in candidates:
        path = os.path.join(MEDIA_DIR, name)
        if os.path.exists(path):
            return path
    return None


def print_summary(summary: dict, video_path: str, order: int):
    label = f"[{order}번 면접 영상]  {os.path.basename(video_path)}"
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  영상 길이         : {summary['duration_sec']:.1f}초")
    print()
    print(f"  [시선]")
    print(f"    정면 응시율      : {summary['focus_rate']}%")
    print(f"    시선 이탈율      : {summary['deviated_gaze_rate']}%")
    print()
    print(f"  [눈 깜빡임]")
    print(f"    총 깜빡임 횟수   : {summary['blink_count']}회")
    print(f"    분당 깜빡임      : {summary['blinks_per_min']}회/분")
    print()
    print(f"  [고개 끄덕임]")
    print(f"    총 끄덕임 횟수   : {summary['nod_count']}회")
    print(f"    분당 끄덕임      : {summary['nod_per_min']}회/분")
    print()
    print(f"  [몸통 흔들림]")
    print(f"    총 흔들림 횟수   : {summary['body_sway_count']}회")
    print(f"    분당 흔들림      : {summary['body_sway_per_min']}회/분")
    dirs = summary.get("body_sway_direction_counts", {})
    if dirs:
        print(f"    방향별 횟수")
        print(f"      좌 : {dirs.get('LR_LEFT', 0)}회  우 : {dirs.get('LR_RIGHT', 0)}회  "
              f"앞 : {dirs.get('FB_FORWARD', 0)}회  뒤 : {dirs.get('FB_BACKWARD', 0)}회")
    print()
    print(f"  [어깨 안정성]")
    print(f"    안정성 비율      : {summary['shoulder_stability']}%")
    print(f"    기울임 발생 횟수 : {summary['shoulder_tilt_count']}회")
    print(f"    안정 프레임      : {summary['shoulder_stable_frames']}  "
          f"기울임 프레임 : {summary['shoulder_tilted_frames']}")
    print()
    print(f"  [미소]")
    print(f"    미소 비율        : {summary['total_smile_rate']}%")
    print("=" * 60)


def debug_shoulder(frame_details: list, summary: dict):
    baseline   = summary.get("interview_baseline", {})
    sh         = baseline.get("shoulder", {})
    base_angle = sh.get("base_shoulder_angle", 0.0)
    tolerance  = sh.get("shoulder_angle_tolerance", 10.0)
    calib_std  = sh.get("calibration_angle_std", 0.0)
    skip_sec   = baseline.get("body_shoulder_baseline_sec", 5.0)

    print()
    print("-" * 60)
    print("  [어깨 안정성 디버그]")
    print(f"    기준 어깨 각도   : {base_angle:.3f}°")
    print(f"    허용 범위        : ±{tolerance:.1f}°  →  [{base_angle - tolerance:.3f}° ~ {base_angle + tolerance:.3f}°]")
    print(f"    baseline 구간    : 0 ~ {skip_sec:.0f}초 (이 구간은 측정 제외)")
    instable_note = "불안정 — 기준값 신뢰 낮음" if calib_std > 4.0 else "안정"
    print(f"    baseline 표준편차: {calib_std:.3f}°  ({instable_note})")

    # baseline 이후이고 어깨가 감지된 프레임만 추출
    measured = [
        d for d in frame_details
        if d.get("timestamp", 0.0) >= skip_sec
        and d.get("shoulder_angle", 0.0) != 0.0
    ]

    if not measured:
        print("    → 측정 구간 내 어깨 감지 프레임 없음")
        print("-" * 60)
        return

    angles  = [d["shoulder_angle"]      for d in measured]
    diffs   = [d["shoulder_angle_diff"] for d in measured]
    stables = [d.get("shoulder_stable", True) for d in measured]

    instable_count = sum(1 for s in stables if not s)
    total_count    = len(measured)

    print()
    print(f"    측정 프레임 수       : {total_count}")
    print(f"    각도 범위            : {min(angles):.3f}° ~ {max(angles):.3f}°")
    print(f"    각도 평균 / 표준편차 : {np.mean(angles):.3f}° / {np.std(angles):.3f}°")
    print(f"    기준대비 차이        : 평균 {np.mean(diffs):.3f}°  최대 {max(diffs):.3f}°")
    print(f"    기울어짐 프레임      : {instable_count} / {total_count}  ({instable_count / total_count * 100:.1f}%)")

    # 각도가 기준범위 밖에 있는 프레임 수 (smoothing 전 raw 기준)
    outside = sum(1 for a in angles if abs(a - base_angle) > tolerance)
    print(f"    raw 각도 범위 초과   : {outside} / {total_count} 프레임  "
          f"(smoothing 전, 참고용)")

    # 기울어짐 구간 타임라인
    print()
    print("    ── 기울어짐 구간 타임라인 ──")
    in_tilt    = False
    tilt_start = 0.0
    segments   = []
    for d in measured:
        t  = d["timestamp"]
        ok = d.get("shoulder_stable", True)
        if not ok and not in_tilt:
            tilt_start = t
            in_tilt    = True
        elif ok and in_tilt:
            segments.append((tilt_start, t))
            in_tilt = False
    if in_tilt:
        segments.append((tilt_start, measured[-1]["timestamp"]))

    if segments:
        for s, e in segments:
            # 해당 구간의 평균 각도도 표시
            seg_angles = [d["shoulder_angle"] for d in measured if s <= d["timestamp"] <= e]
            avg_a = np.mean(seg_angles) if seg_angles else 0.0
            print(f"      {s:.2f}s ~ {e:.2f}s  (지속 {e - s:.2f}초, 구간 평균각도 {avg_a:.3f}°)")
    else:
        print("      기울어짐 구간 없음 (전 구간 안정)")

    # 차이가 큰 상위 10 프레임 목록
    print()
    print("    ── 기준각 대비 차이 상위 10 프레임 ──")
    print(f"      {'시간':>7}  {'실제각도':>9}  {'기준대비차':>8}  {'판정':>6}")
    top = sorted(measured, key=lambda d: d["shoulder_angle_diff"], reverse=True)[:10]
    for d in top:
        state = "기울어짐" if not d.get("shoulder_stable", True) else "  안정  "
        print(
            f"      {d['timestamp']:>6.2f}s"
            f"  {d['shoulder_angle']:>9.3f}°"
            f"  {d['shoulder_angle_diff']:>7.3f}°"
            f"  {state}"
        )

    # ASCII 타임라인 그래프
    print()
    print("    ── 시간대별 상태 (▲=기울어짐  ·=안정  _=미감지) ──")
    BUCKETS  = 40
    t_start  = measured[0]["timestamp"]
    t_end    = measured[-1]["timestamp"]
    duration = t_end - t_start
    if duration > 0:
        bucket_sec = duration / BUCKETS
        bar = ""
        for i in range(BUCKETS):
            t_lo    = t_start + i * bucket_sec
            t_hi    = t_lo + bucket_sec
            bucket  = [d for d in measured if t_lo <= d["timestamp"] < t_hi]
            if not bucket:
                bar += "_"
            elif any(not d.get("shoulder_stable", True) for d in bucket):
                bar += "▲"
            else:
                bar += "·"
        print(f"      {t_start:.1f}s |{bar}| {t_end:.1f}s")
    print("-" * 60)


def print_json_summary(summary: dict):
    clean = {k: v for k, v in summary.items() if k != "interview_baseline"}
    print(json.dumps(clean, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="서버 없이 행동 분석 결과를 확인합니다."
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="초기 세팅(캘리브레이션) 영상 경로 (기본값: media/calib.mp4)",
    )
    parser.add_argument(
        "--interview",
        nargs="+",
        default=None,
        help="면접 영상 경로 (여러 개 가능, 기본값: media/interview.mp4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="결과를 JSON 형태로도 출력",
    )
    parser.add_argument(
        "--debug-shoulder",
        action="store_true",
        help="어깨 안정성 상세 디버그 출력 (각도 변화, 기울어짐 구간, 타임라인)",
    )
    args = parser.parse_args()

    # --- 초기 세팅 영상 경로 결정 ---
    calib_path = args.calib or find_default_video(DEFAULT_CALIB_NAMES)
    if calib_path is None:
        print("[ERROR] 초기 세팅 영상을 찾을 수 없습니다.")
        print(f"        media/ 폴더에 {DEFAULT_CALIB_NAMES} 중 하나를 넣거나")
        print("        --calib 옵션으로 경로를 직접 지정하세요.")
        sys.exit(1)

    if not os.path.exists(calib_path):
        print(f"[ERROR] 초기 세팅 영상이 없습니다: {calib_path}")
        sys.exit(1)

    ext = os.path.splitext(calib_path)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        print(f"[ERROR] 지원하지 않는 파일 형식입니다: {calib_path}")
        sys.exit(1)

    # --- 면접 영상 경로 결정 ---
    interview_paths = args.interview
    if interview_paths is None:
        default_interview = find_default_video(DEFAULT_INTERVIEW_NAMES)
        if default_interview is None:
            print("[ERROR] 면접 영상을 찾을 수 없습니다.")
            print(f"        media/ 폴더에 {DEFAULT_INTERVIEW_NAMES} 중 하나를 넣거나")
            print("        --interview 옵션으로 경로를 직접 지정하세요.")
            sys.exit(1)
        interview_paths = [default_interview]

    for path in interview_paths:
        if not os.path.exists(path):
            print(f"[ERROR] 면접 영상이 없습니다: {path}")
            sys.exit(1)
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            print(f"[ERROR] 지원하지 않는 파일 형식입니다: {path}")
            sys.exit(1)

    # --- import (Django 없이 직접 사용) ---
    sys.path.insert(0, os.path.dirname(__file__))
    from behavior.analysis_logic import run_calibration, analyze_behavior_video

    # --- 캘리브레이션 ---
    print()
    print("=" * 60)
    print(f"  초기 세팅 영상 분석 중...  ({os.path.basename(calib_path)})")
    print("=" * 60)

    try:
        config = run_calibration(calib_path)
    except Exception as e:
        print(f"[ERROR] 캘리브레이션 실패: {e}")
        sys.exit(1)

    print("  캘리브레이션 완료")
    print(f"    EAR 임계값       : {config['ear_threshold']:.4f}")
    print(f"    기준 고개 방향   : {config['base_head_turn']:.4f}")
    print(f"    기준 시선 비율   : {config['base_gaze_ratio']:.4f}")

    # --- 면접 영상 분석 ---
    print()
    print(f"  면접 영상 {len(interview_paths)}개 분석 시작")

    all_summaries = []

    for order, video_path in enumerate(interview_paths, start=1):
        print()
        print(f"  [{order}/{len(interview_paths)}] {os.path.basename(video_path)} 분석 중...")

        try:
            frame_details, summary = analyze_behavior_video(video_path, config=config)
        except Exception as e:
            print(f"  [ERROR] 분석 실패: {e}")
            continue

        all_summaries.append((video_path, frame_details, summary))
        print_summary(summary, video_path, order)

        if args.debug_shoulder:
            debug_shoulder(frame_details, summary)

        if args.json:
            print()
            print("  [JSON 원본]")
            print_json_summary(summary)

    if not all_summaries:
        print("\n[ERROR] 분석된 영상이 없습니다.")
        sys.exit(1)

    # --- 여러 영상이 있을 때 전체 집계 ---
    if len(all_summaries) > 1:
        total_duration = sum(s["duration_sec"] for _, _fd, s in all_summaries)
        total_duration_min = total_duration / 60 if total_duration > 0 else 1.0

        total_blinks    = sum(s["blink_count"]       for _, _fd, s in all_summaries)
        total_nods      = sum(s["nod_count"]          for _, _fd, s in all_summaries)
        total_sways     = sum(s["body_sway_count"]    for _, _fd, s in all_summaries)
        avg_focus       = sum(s["focus_rate"]         for _, _fd, s in all_summaries) / len(all_summaries)
        avg_smile       = sum(s["total_smile_rate"]   for _, _fd, s in all_summaries) / len(all_summaries)
        avg_stability   = sum(s["shoulder_stability"] for _, _fd, s in all_summaries) / len(all_summaries)

        print()
        print("=" * 60)
        print(f"  전체 집계  ({len(all_summaries)}개 영상)")
        print("=" * 60)
        print(f"  총 영상 길이       : {total_duration:.1f}초")
        print(f"  평균 정면 응시율   : {avg_focus:.1f}%")
        print(f"  총 깜빡임          : {total_blinks}회  ({total_blinks / total_duration_min:.1f}회/분)")
        print(f"  총 고개 끄덕임     : {total_nods}회  ({total_nods / total_duration_min:.1f}회/분)")
        print(f"  총 몸통 흔들림     : {total_sways}회  ({total_sways / total_duration_min:.1f}회/분)")
        print(f"  평균 어깨 안정성   : {avg_stability:.1f}%")
        print(f"  평균 미소 비율     : {avg_smile:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
