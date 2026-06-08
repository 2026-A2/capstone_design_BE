import argparse
import os

import ffmpeg

from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers
from speech.aggregator import aggregate_results

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio(video_path: str) -> str:
    wav_path = video_path + "_audio.wav"
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


def analyze_file(file_path: str, order: int) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    extracted_path = None

    try:
        if ext in VIDEO_EXTENSIONS:
            print(f"  오디오 추출 중...")
            extracted_path = extract_audio(file_path)
            analyze_path = extracted_path
        else:
            analyze_path = file_path

        print(f"  STT 분석 중...")
        result = analyze(analyze_path)
        segments = result.pop("segments", None)

        print(f"  음량 분석 중...")
        volume = analyze_volume(analyze_path, segments=segments)

        print(f"  습관어 감지 중...")
        filler = detect_fillers(analyze_path)

        result["volume"] = volume
        result["filler"] = filler
        result["order"] = order
        result["file_name"] = os.path.basename(file_path)

        return result

    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)


def print_summary(summary: dict, video_count: int):
    print()
    print("=" * 60)
    print(f"  전체 집계 결과  ({video_count}개 영상 평균)")
    print("=" * 60)
    print(f"  평균 발화 속도  : {summary['avg_spm']} SPM  ({summary['pace']})")
    print(f"  평균 음량      : {summary['avg_db']} dB  ({summary['volume_level']})")
    print(f"  최대 음량      : {summary['max_db']} dB")
    print(f"  최소 음량      : {summary['min_db']} dB")
    print(f"  음량 표준편차  : {summary['avg_std_db']} dB")
    print(f"  총 습관어      : {summary['total_filler_count']}회")
    frequent = ", ".join(f'"{f}"' for f in summary["frequent_fillers"]) or "없음"
    print(f"  자주 쓴 습관어 : {frequent}")
    print(f"  총 침묵 구간   : {summary['total_silence_count']}회")
    print(f"  평균 침묵 길이 : {summary['avg_silence_duration']}초")
    print("=" * 60)


def print_individual(result: dict):
    order = result["order"]
    file_name = result["file_name"]

    print()
    print(f"  [{order}번 영상]  {file_name}")
    print("-" * 60)

    print(f"  전사 텍스트:")
    print(f"    {result['transcript']}")
    print()

    silences = result["silences"]
    if silences:
        print(f"  침묵 구간 ({len(silences)}회):")
        for s in silences:
            print(f"    {s['start']:.2f}s ~ {s['end']:.2f}s  ({s['duration']:.2f}s)")
    else:
        print(f"  침묵 구간: 없음")
    print()

    fillers = result["filler"]["fillers"]
    if fillers:
        print(f"  습관어 ({len(fillers)}회):")
        for f in fillers:
            print(f"    {f['start']:.2f}s ~ {f['end']:.2f}s  \"{f['type']}\"")
    else:
        print(f"  습관어: 없음")
    print()

    trailing = result["volume"]["trailing_off"]
    if trailing:
        print(f"  말끝 흐림 ({len(trailing)}회):")
        for t in trailing:
            print(f"    {t['start']:.2f}s ~ {t['end']:.2f}s  (감소율 {t['감소율']}%)")
    else:
        print(f"  말끝 흐림: 없음")


def main():
    parser = argparse.ArgumentParser(
        description="면접 영상 n개를 분석해 항목별 평균 결과를 출력합니다."
    )
    parser.add_argument("files", nargs="+", help="분석할 영상/음성 파일 경로 (여러 개 가능)")
    args = parser.parse_args()

    for path in args.files:
        if not os.path.exists(path):
            print(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[ERROR] 지원하지 않는 형식입니다: {path}")
            print(f"        지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return

    print(f"\n[면접 음성 분석]  총 {len(args.files)}개 영상")
    print("=" * 60)

    all_results = []
    for order, file_path in enumerate(args.files, start=1):
        print(f"\n{order}번 영상 분석 중...  ({os.path.basename(file_path)})")
        result = analyze_file(file_path, order=order)
        all_results.append(result)

    summary = aggregate_results(all_results)

    print_summary(summary, video_count=len(all_results))

    print()
    print("=" * 60)
    print("  영상별 상세 결과  ")
    print("=" * 60)
    for result in all_results:
        print_individual(result)


if __name__ == "__main__":
    main()
