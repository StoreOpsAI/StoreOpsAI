"""질문 시점 VLM 해석: 사건 대표 이미지를 Qwen VLM(OpenAI 호환 API)에 보내 장면을 한국어로 설명받습니다.

사건이 난 순간에는 VLM을 부르지 않고, 점주가 지난 사건을 물을 때만 호출합니다.
표준 라이브러리와 Pillow만 사용합니다. 이미지는 VLM 서버의 입력 길이 한도에 맞게 줄여서 보냅니다.
"""

import base64
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


class VlmUnavailableError(Exception):
    """VLM 서버를 쓸 수 없을 때(주소 없음, 연결 실패, 응답 오류)입니다."""


PROMPT = (
    "무인 매장 CCTV 대표 이미지 {count}장(처음·가운데·끝 순서)입니다. "
    "분류 모델이 '{category}' 의심으로 표시한 사건입니다. "
    "이미지에서 실제로 보이는 사람의 위치·자세·행동과 물건의 상태만 한국어 2~3문장으로 설명하세요. "
    "분류 모델의 판정은 사실로 단정하지 말고, 이미지에서 확인되지 않으면 '이미지만으로는 확인되지 않음'이라고 쓰세요. "
    "보이지 않는 것은 추측하지 마세요. 사람의 신원은 묻거나 추측하지 마세요."
)


def _resize_to_data_url(path: Path, max_pixels: int, quality: int) -> str:
    """이미지를 max_pixels 이하로 줄여 JPEG data URL로 만듭니다(원본 파일은 바꾸지 않음)."""

    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def describe_images(image_paths: list[Path], category_ko: str) -> dict:
    """대표 이미지들을 VLM에 보내 설명 문장을 받아 {'text', 'model_name'}으로 돌려줍니다."""

    base_url = os.getenv('QWEN_VLM_BASE_URL', '').rstrip('/')
    if not base_url:
        raise VlmUnavailableError('QWEN_VLM_BASE_URL이 설정되지 않았습니다.')
    if not image_paths:
        raise VlmUnavailableError('해석할 대표 이미지가 없습니다.')
    model = os.getenv('QWEN_VLM_MODEL', 'Qwen/Qwen3-VL-8B-Instruct-FP8')
    max_pixels = int(os.getenv('QWEN_VLM_IMAGE_MAX_PIXELS', '147456'))
    quality = int(os.getenv('QWEN_VLM_IMAGE_JPEG_QUALITY', '85'))
    content = [{'type': 'text', 'text': PROMPT.format(count=len(image_paths), category=category_ko)}]
    content += [
        {'type': 'image_url', 'image_url': {'url': _resize_to_data_url(path, max_pixels, quality)}}
        for path in image_paths
    ]
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'temperature': 0.1,
        'max_tokens': int(os.getenv('QWEN_VLM_MAX_TOKENS', '300')),
    }, ensure_ascii=False).encode('utf-8')
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    api_key = os.getenv('QWEN_VLM_API_KEY', '')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = Request(base_url + '/chat/completions', data=body, headers=headers, method='POST')
    try:
        with urlopen(request, timeout=float(os.getenv('QWEN_VLM_TIMEOUT_SEC', '60'))) as response:
            raw = json.loads(response.read().decode('utf-8'))
        text = raw['choices'][0]['message']['content'].strip()
    except (HTTPError, URLError, OSError, KeyError, IndexError, ValueError) as error:
        raise VlmUnavailableError(f'VLM 응답을 받지 못했습니다: {error}') from error
    if not text:
        raise VlmUnavailableError('VLM이 빈 응답을 돌려주었습니다.')
    return {'text': text, 'model_name': model}
