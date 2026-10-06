from storeops_ai.pipelines import path2_connection


def test_disconnect_once_after_threshold_and_alert_on_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(path2_connection, "EVENT_DIR", tmp_path)
    sent_alerts = []
    monkeypatch.setattr(
        path2_connection,
        "send_first_alert",
        lambda event: sent_alerts.append(event) or True,
    )
    monitor = path2_connection.CameraConnectionMonitor("CAM-01", disconnect_threshold=60)

    monitor.frame_received(100)
    assert monitor.check(160) is None
    disconnect_event = monitor.check(190)

    assert disconnect_event is not None
    assert disconnect_event.disconnect_seconds == 90
    assert monitor.check(200) is None
    assert len(sent_alerts) == 1

    monitor.frame_received(205)

    assert monitor.state()["connected"] is True
    assert len(sent_alerts) == 2
    assert sent_alerts[1].event_type == "카메라재연결"
    assert sent_alerts[1].disconnect_seconds == 105
    assert len(list(tmp_path.glob("E*.json"))) == 2


def test_reconnect_detects_outage_before_periodic_check(tmp_path, monkeypatch):
    monkeypatch.setattr(path2_connection, "EVENT_DIR", tmp_path)
    sent_alerts = []
    monkeypatch.setattr(
        path2_connection,
        "send_first_alert",
        lambda event: sent_alerts.append(event) or True,
    )
    monitor = path2_connection.CameraConnectionMonitor("CAM-02", disconnect_threshold=60)

    monitor.frame_received(100)
    monitor.frame_received(190)

    assert monitor.state()["connected"] is True
    assert [event.event_type for event in sent_alerts] == ["카메라끊김", "카메라재연결"]
    assert sent_alerts[0].disconnect_seconds == 90
    assert sent_alerts[1].disconnect_seconds == 90