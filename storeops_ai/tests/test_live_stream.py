"""실시간 화면 서버 자체 검사(utils/live_stream.py). 실행: python -m unittest discover -s tests -p test_live_stream.py"""
import unittest

from utils.live_stream import _self_check


class TestLiveStream(unittest.TestCase):
    def test_self_check(self):
        _self_check()


if __name__ == "__main__":
    unittest.main()
