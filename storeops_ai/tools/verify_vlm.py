"""대표 이미지 세 장으로 Qwen3-VL 실제 추론을 확인하는 도구."""

import argparse
import json
import sys
from pathlib import Path

from models.vlm_analyzer import VLMAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL 이미지 추론 연결 확인")
    parser.add_argument(
        "--images",
        nargs=3,
        metavar=("FIRST", "MIDDLE", "LAST"),
        required=True,
        help="사건 시작/중간/끝 대표 이미지 파일 3개",
    )
    parser.add_argument("--event-id", default="VLM-TEST")
    parser.add_argument("--camera-id", default="CAM-01")
    args = parser.parse_args()

    image_paths = [str(Path(path)) for path in args.images]
    missing_paths = [path for path in image_paths if not Path(path).is_file()]
    if missing_paths:
        print(f"이미지 파일을 찾을 수 없습니다: {', '.join(missing_paths)}", file=sys.stderr)
        return 2

    analyzer = VLMAnalyzer()
    try:
        result = analyzer.analyze(
            image_paths,
            {"event_id": args.event_id, "camera_id": args.camera_id},
        )
    except Exception as error:
        print(f"Qwen 이미지 추론 실패: {error}", file=sys.stderr)
        return 1
    finally:
        analyzer.close()

    if result.get("status") == "VLM_NOT_CONFIGURED":
        print("QWEN_VLM_BASE_URL이 설정되지 않았습니다.", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
