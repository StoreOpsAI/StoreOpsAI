"""점주 대시보드 실시간 화면: 탐지 화면(데모 창과 같은 화면)의 최신 JPEG 한 장을 토큰이 맞는 요청에만 돌려준다.

- 카메라는 탐지 프로그램만 연다(브라우저와 웹캠을 나눠 쓰지 않는다). 백엔드가 점주 로그인을 확인한 뒤 이 주소를 대신 부른다.
- GET /live/frame?after=N: 번호 N과 다른(새) 화면이 생길 때까지 최대 wait_sec 기다렸다가 JPEG(헤더 X-Frame-Seq)를 준다. 없으면 204.
  브라우저는 받은 번호로 다시 요청한다(롱 폴링). 화면 속도는 탐지 처리 속도를 따른다.
- 인증: Authorization: Bearer <EVENT_INGEST_TOKEN>. 토큰이 비어 있으면 서버를 열지 않는다(매장 화면이 인증 없이 열리지 않게).
"""
from __future__ import annotations

import hmac
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


class LiveStream:
    def __init__(self, token: str, wait_sec: float):
        if not token:
            raise ValueError("실시간 화면에는 EVENT_INGEST_TOKEN이 필요합니다(인증 없이 매장 화면을 열지 않음).")
        self.token, self.wait_sec = token, wait_sec
        self.seq, self.jpeg = 0, b""
        self._cv = threading.Condition()

    def publish(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return
        with self._cv:
            self.seq, self.jpeg = self.seq + 1, buf.tobytes()
            self._cv.notify_all()

    def newer_than(self, after: int) -> tuple[int, bytes] | None:
        with self._cv:
            fresh = lambda: self.jpeg != b"" and self.seq != after   # 탐지를 다시 켜 번호가 처음부터 시작해도 멈추지 않게 '다르면 새 화면'
            self._cv.wait_for(fresh, timeout=self.wait_sec)
            return (self.seq, self.jpeg) if fresh() else None

    def authorized(self, header: str | None) -> bool:
        received = (header or "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(received.encode(), self.token.encode())

    def serve(self, host: str, port: int) -> ThreadingHTTPServer:
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                if url.path != "/live/frame":
                    return self._reply(404)
                if not stream.authorized(self.headers.get("Authorization")):
                    return self._reply(401)
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    return self._reply(422)
                got = stream.newer_than(after)
                if got is None:
                    return self._reply(204)
                self._reply(200, got[1], {"Content-Type": "image/jpeg", "X-Frame-Seq": str(got[0])})

            def _reply(self, code, body=b"", headers=None):
                self.send_response(code)
                for k, v in {"Cache-Control": "no-store", **(headers or {})}.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):   # 요청마다 콘솔에 찍지 않는다
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


def _self_check() -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    try:
        LiveStream("", 1.0)
        raise AssertionError("빈 토큰이면 열지 않아야 한다")
    except ValueError:
        pass
    s = LiveStream("secret", 0.2)
    server = s.serve("127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/live/frame"
    get = lambda q, tok="secret": urlopen(Request(url + q, headers={"Authorization": f"Bearer {tok}"}), timeout=5)
    try:
        get("?after=0", "wrong")
        raise AssertionError("틀린 토큰은 401")
    except HTTPError as e:
        assert e.code == 401
    assert get("?after=0").status == 204                     # 아직 화면 없음
    s.publish(np.zeros((8, 8, 3), np.uint8))
    r = get("?after=0")
    assert r.status == 200 and r.headers["X-Frame-Seq"] == "1" and r.read()[:2] == b"\xff\xd8"
    assert get("?after=1").status == 204                     # 새 화면이 없으면 기다렸다가 204
    assert get("?after=99").headers["X-Frame-Seq"] == "1"   # 탐지를 다시 켜 번호가 줄어도 지금 화면을 준다
    server.shutdown()
    print("live_stream self-check ok")


if __name__ == "__main__":
    _self_check()
