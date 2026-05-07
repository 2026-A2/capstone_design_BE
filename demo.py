import argparse
import json
import os
from datetime import datetime
from speech.input_handler import validate_wav
from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers


def print_result(result: dict):
    print()
    print("=" * 55)
    print("  전사 결과")
    print("=" * 55)
    print(result["transcript"])
    print()
    print("=" * 55)
    print("  발화 속도 분석")
    print("=" * 55)
    print(f"  전체 길이      : {result['duration_sec']:.2f}초")
    print(f"  음절 수        : {result['syllable_count']}음절")
    print(f"  발화 속도      : {result['spm']} SPM")
    print(f"  속도 평가      : {result['pace']}")
    print("=" * 55)
    print()
    print("=" * 55)
    print("  침묵 감지  (3초 이상 무발화 구간)")
    print("=" * 55)
    silences = result["silences"]
    if not silences:
        print("  감지된 침묵 없음")
    else:
        print(f"  침묵 횟수      : {len(silences)}회")
        for i, s in enumerate(silences, 1):
            print(f"  [{i}] {s['start']:.2f}s ~ {s['end']:.2f}s  ({s['duration']:.2f}s)")
    print("=" * 55)
    print()
    print("=" * 55)
    print("  음량 분석")
    print("=" * 55)
    vol = result["volume"]
    print(f"  전체 평균 음량  : {vol['avg_db']} dB")
    print(f"  최대 음량      : {vol['max_db']} dB")
    print(f"  최소 음량      : {vol['min_db']} dB")
    print(f"  음량 변동폭    : {vol['std_db']} dB")
    print(f"  음량 수준      : {vol['volume_level']}")
    print("=" * 55)
    print()
    print("=" * 55)
    print("  말끝 흐림 감지  (문장 끝 0.5초 RMS 80% 이상 감소)")
    print("=" * 55)
    trailing = vol["trailing_off"]
    print(f"  말끝 흐림 횟수 : {len(trailing)}회")
    print("=" * 55)
    print()
    print("=" * 55)
    print("  필러 감지")
    print("=" * 55)
    filler = result["filler"]
    frequent = ", ".join(f'"{f}"' for f in filler["frequent_fillers"]) or "없음"
    print(f"  필러 사용 횟수  : {filler['filler_count']}회")
    print(f"  자주 사용한 필러 : {frequent}")
    print("=" * 55)
    print()


def save_result(result: dict, audio_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join("result", f"{filename}_{timestamp}.json")

    volume = dict(result.get("volume", {}))
    volume_timeline = volume.pop("volume_timeline", [])

    data = {
        "audio_file": audio_path,
        "analyzed_at": datetime.now().isoformat(),
        **{k: v for k, v in result.items() if k != "volume"},
        "volume": volume,
        "volume_timeline": volume_timeline,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse speech rate and transcribe a WAV file using OpenAI Whisper."
    )
    parser.add_argument("wav_file", help="Path to the input .wav file")
    args = parser.parse_args()

    print("\n[음성 분석기]")
    print("-" * 55)

    wav_path = validate_wav(args.wav_file)

    print("\nWhisper 전사 중...")
    result = analyze(wav_path)

    print("\n음량 분석 중...")
    result["volume"] = analyze_volume(wav_path, segments=result.get("segments"))
    del result["segments"]

    print("\n필러 감지 중...")
    result["filler"] = detect_fillers(wav_path)

    print_result(result)
    save_result(result, args.wav_file)


if __name__ == "__main__":
    main()
