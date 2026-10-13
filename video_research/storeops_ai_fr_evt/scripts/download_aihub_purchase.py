"""AI Hub 구매행동 데이터(71549) 다운로드. aihubshell과 같은 API를 쓰고 표준 라이브러리만 사용한다.

API 키는 환경변수 AIHUB_APIKEY 또는 숨김 입력으로 받는다(명령줄·셸 기록에 남지 않음).
받은 tar를 풀고 .partN 조각을 합친 뒤 tar는 지운다. 끊기면 다시 실행해 이어받는다.

사용 (WSL):
    python3 storeops_ai_fr_evt/scripts/download_aihub_purchase.py               # 라벨 5개 + 반품 영상
    python3 storeops_ai_fr_evt/scripts/download_aihub_purchase.py --keys 546978  # 구매 영상만
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

DATASET_KEY = "71549"
DOWNLOAD_URL = "https://api.aihub.or.kr/down/0.6/{dataset}.do?fileSn={keys}"
LABEL_KEYS = "546982,546983,546984,546985,546986"  # Validation 라벨: 선택·시험·구매·반품·비교
VIDEO_KEYS = "546979"                                # Validation 영상: 반품 (14.5GB)
CHUNK = 8 * 1024 * 1024


def download(url, api_key, target):
    """target에 이어받는다. 서버가 이어받기를 무시하면(200) 처음부터 다시 받는다."""
    done = target.stat().st_size if target.exists() else 0
    headers = {"apikey": api_key}
    if done:
        headers["Range"] = f"bytes={done}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416:  # 이미 다 받음
            return
        body = error.read(2000).decode("utf-8", "replace").replace(api_key, "<API 키>")  # 서버가 헤더를 되돌려줌
        message = re.search(r"<b>Message</b>(.*?)</p>", body, re.S)
        raise SystemExit(f"다운로드 실패 HTTP {error.code}: {(message.group(1) if message else body)[:300]}") from None
    with response:
        mode = "ab" if response.status == 206 else "wb"
        done = done if mode == "ab" else 0
        total = int(response.headers.get("Content-Length", 0)) + done
        with target.open(mode) as file:
            while chunk := response.read(CHUNK):
                file.write(chunk)
                done += len(chunk)
                percent = f" {done / total:6.1%}" if total else ""
                print(f"\r{target.name}: {done / 2**30:7.2f} GB{percent}", end="", flush=True)
    print()


def extract(tar_path, dest):
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not (dest / member.name).resolve().is_relative_to(dest.resolve()):
                raise SystemExit(f"tar 안에 폴더 밖을 가리키는 경로가 있습니다: {member.name}")
        archive.extractall(dest)


def merge_parts(dest):
    """name.zip.part0, part1, ... 을 숫자 순서로 합쳐 name.zip을 만든다."""
    groups = {}
    for part in dest.rglob("*.part*"):
        match = re.fullmatch(r"(.*)\.part(\d+)", part.name)
        if match:
            groups.setdefault(part.with_name(match.group(1)), []).append((int(match.group(2)), part))
    for merged, parts in groups.items():
        print(f"병합: {merged.name} ({len(parts)}조각)")
        with merged.open("wb") as out:
            for _, part in sorted(parts):
                with part.open("rb") as source:
                    while chunk := source.read(CHUNK):
                        out.write(chunk)
        for _, part in parts:
            part.unlink()


def fetch(keys, api_key, dest):
    tar_path = dest / f"download_{keys.replace(',', '_')}.tar"
    download(DOWNLOAD_URL.format(dataset=DATASET_KEY, keys=keys), api_key, tar_path)
    if not tarfile.is_tarfile(tar_path):
        raise SystemExit(f"받은 파일이 tar가 아닙니다(키·승인 확인): {tar_path.read_bytes()[:500]!r}")
    extract(tar_path, dest)
    merge_parts(dest)
    tar_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="AI Hub 구매행동(71549) 다운로드")
    parser.add_argument("--keys", nargs="+", default=[LABEL_KEYS, VIDEO_KEYS],
                        help="파일키 묶음(쉼표 구분). 묶음마다 따로 받는다.")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[2] / "구매행동")
    args = parser.parse_args()

    raw = os.environ.get("AIHUB_APIKEY") or getpass.getpass("AI Hub API 키 (붙여넣고 바로 엔터, 방향키 금지): ")
    api_key = re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\s", "", raw)  # 방향키 등 터미널 제어 문자 제거
    if not re.fullmatch(r"[0-9A-Za-z-]{16,64}", api_key):
        sys.exit("키 형식이 올바르지 않습니다(영문·숫자·하이픈만). 다시 붙여넣어 주세요.")  # 키 내용은 출력하지 않는다
    args.dest.mkdir(parents=True, exist_ok=True)
    for keys in args.keys:
        fetch(keys, api_key, args.dest)
    print(f"완료: {args.dest}")


if __name__ == "__main__":
    main()
