import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from config.config import ALERT_WEBHOOK_TIMEOUT_SEC, ALERT_WEBHOOK_URL, VLM_RESULT_WEBHOOK_URL
logger = logging.getLogger("storeops_ai")
logging.basicConfig(level=logging.INFO)


def send_first_alert(event):
    """FR-EVT-07/11: 설정된 HTTP 웹훅(StoreOpsAI 백엔드 사건 등록 API)으로 즉시 전송한다."""
    if not ALERT_WEBHOOK_URL:
        event.alert_status = "NOT_CONFIGURED"
        event.alert_error = "ALERT_WEBHOOK_URL is not configured"
        logger.warning("[FIRST ALERT NOT CONFIGURED] event=%s", event.event_id)
        return False
    # backend가 같은 output 볼륨을 공유하므로 경로는 상대 경로 그대로 보낸다(백엔드 /clip 라우터가 해석).
    body = json.dumps(event.to_dict(), ensure_ascii=False).encode("utf-8")
    ingest_token = os.getenv("EVENT_INGEST_TOKEN", "")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if ingest_token:
        headers["Authorization"] = f"Bearer {ingest_token}"
    request = Request(ALERT_WEBHOOK_URL, data=body, method="POST",
                      headers=headers)
    try:
        with urlopen(request, timeout=ALERT_WEBHOOK_TIMEOUT_SEC) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"webhook returned HTTP {response.status}")
            response_body = response.read()
            if response_body:
                try:
                    event.backend_event_id = json.loads(response_body.decode("utf-8")).get("event_id")
                except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                    logger.warning("[EVENT ID MISSING] event=%s", event.event_id)
        event.alert_status = "SENT"
        event.alert_error = None
        logger.info("[FIRST ALERT SENT] event=%s", event.event_id)
        return True
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        event.alert_status = "FAILED"
        # HTTPError는 str()만으로 422 검증 사유가 안 보이므로 응답 본문을 함께 남긴다.
        detail = exc.read().decode("utf-8", "replace") if isinstance(exc, HTTPError) else ""
        event.alert_error = f"{exc}" + (f" | {detail}" if detail else "")
        logger.exception("[FIRST ALERT FAILED] event=%s detail=%s", event.event_id, detail)
        return False


def send_vlm_result(event, status, result=None, error=None):
    """비동기 VLM 결과를 사건 등록 API가 반환한 ID로 백엔드에 전달한다."""
    if not VLM_RESULT_WEBHOOK_URL or not event.backend_event_id:
        logger.warning("[VLM RESULT NOT CONFIGURED] event=%s", event.event_id)
        return False
    result = result or {}
    payload = {
        "status": status,
        "observation": result.get("observation"),
        "uncertain_points": result.get("uncertain_points"),
        "owner_actions": result.get("owner_actions", []),
        "model_name": result.get("model_name"),
        "error": error,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    ingest_token = os.getenv("EVENT_INGEST_TOKEN", "")
    if ingest_token:
        headers["Authorization"] = f"Bearer {ingest_token}"
    url = VLM_RESULT_WEBHOOK_URL.format(event_id=event.backend_event_id)
    request = Request(url, data=body, method="POST", headers=headers)
    try:
        with urlopen(request, timeout=ALERT_WEBHOOK_TIMEOUT_SEC) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"webhook returned HTTP {response.status}")
        return True
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        logger.exception("[VLM RESULT CALLBACK FAILED] event=%s error=%s", event.event_id, exc)
        return False
