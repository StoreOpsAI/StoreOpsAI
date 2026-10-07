"""규정 질문의 제한적인 오타·띄어쓰기 보정. 번호와 수치는 변경하지 않는다."""
import re

# 의미가 명확한 업무 용어만 보정한다. 임의의 유사 단어로 바꾸지 않는다.
_TYPOS = {"카매라": "카메라", "카메러": "카메라", "보괸": "보관",
          "보간기간": "보관기간", "기갼": "기간", "점검규졍": "점검규정",
          "연겷": "연결", "끈김": "끊김", "알림설졍": "알림설정"}
_TERMS = sorted({"카메라", "영상", "보관", "기간", "점검", "규정", "연결", "끊김",
                 "알림", "기록", "확인", "절차", "운영", "매장", "상시", "정책",
                 "발주", "초안", "사건", "쓰러짐", "며칠", "시간", "주기"}, key=len, reverse=True)
_TERM_PATTERN = re.compile("|".join(map(re.escape, _TERMS)))


def normalize_question(question: str) -> str:
    def normalize_run(match):
        run = match.group()
        for typo, corrected in _TYPOS.items():
            run = run.replace(typo, corrected)
        # 붙여 쓴 한국어 안에서 업무 용어를 분리한다.
        return _TERM_PATTERN.sub(lambda term: " " + term.group() + " ", run)

    return re.sub(r"\s+", " ", re.sub(r"[가-힣]+", normalize_run, question)).strip()
