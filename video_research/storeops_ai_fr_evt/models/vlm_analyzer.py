"""FR-EVT-13~16 VLM 어댑터.

실제 VLM 모델/API는 아직 넣지 않는다. 대신
- 대표 이미지 3장(first/middle/last) 요청 payload 생성
- 3개 필드 검증
- 비동기 실행
을 구현한다.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from config.config import VLM_MAX_WORKERS


class VLMAnalyzer:
    REQUIRED_KEYS = ("observation", "uncertain_points", "owner_actions")

    def __init__(self, transport: Optional[Callable[[List[str], Dict[str, Any]], Dict[str, Any]]] = None):
        self.transport = transport
        self.executor = ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS)

    @classmethod
    def build_request(cls, image_paths: List[str], event_context: Dict[str, Any]) -> Dict[str, Any]:
        if len(image_paths) != 3:
            raise ValueError(f"VLM 입력 이미지는 처음/가운데/끝 3장이 필요합니다. 현재 {len(image_paths)}장")
        return {
            "images": {
                "first": image_paths[0],
                "middle": image_paths[1],
                "last": image_paths[2],
            },
            "prompt": {
                "observation": "관찰한 내용을 작성",
                "uncertain_points": "확실하지 않은 점을 작성",
                "owner_actions": "점주가 확인할 일 3가지를 작성",
            },
            "event_context": event_context,
        }

    def analyze(self, image_paths: List[str], event_context: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.build_request(image_paths, event_context)
        if self.transport is None:
            return {
                "observation": "",
                "uncertain_points": "",
                "owner_actions": [],
                "status": "VLM_NOT_CONFIGURED",
                "request": payload,
            }
        result = self.transport(image_paths, event_context)
        if not self.validate(result):
            raise ValueError("VLM 결과에 observation/uncertain_points/owner_actions 중 누락된 항목이 있습니다.")
        return result

    def analyze_async(self, image_paths: List[str], event_context: Dict[str, Any]):
        # FR-EVT-15: 사건 저장/알림과 분리된 작업
        return self.executor.submit(self.analyze, image_paths, event_context)

    @classmethod
    def validate(cls, result: Any) -> bool:
        return isinstance(result, dict) and all(k in result for k in cls.REQUIRED_KEYS)
