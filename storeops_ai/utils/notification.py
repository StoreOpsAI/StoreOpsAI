import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from config.config import ALERT_WEBHOOK_TIMEOUT_SEC, ALERT_WEBHOOK_URL
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
