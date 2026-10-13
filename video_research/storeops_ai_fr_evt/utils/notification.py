import logging
logger = logging.getLogger("storeops_ai")
logging.basicConfig(level=logging.INFO)


def send_first_alert(event):
    """FR-EVT-07/11: 실제 외부 알림 대신 즉시 호출되는 알림 함수."""
    logger.info(
        "[FIRST ALERT] event=%s camera=%s type=%s category=%s confidence=%s",
        event.event_id, event.camera_id, event.event_type, event.category, event.confidence,
    )
    return True
