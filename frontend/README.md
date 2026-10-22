<p align="center">
	<img src="https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white" alt="React 19">
	<img src="https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white" alt="Vite 8">
</p>

# StoreOps AI 프런트엔드

StoreOps AI 점주용 웹 애플리케이션입니다. React 19와 Vite를 사용하며, 로그인한 점주가 탐지 PC의 실시간 카메라 화면과 최근 사건을 함께 확인하고 사건을 처리하며 발주 초안·매장 질문을 관리합니다.

## 요구 사항

- Node.js 및 npm
- 실행 중인 StoreOps AI 백엔드

전체 시스템 구성은 [프로젝트 README](../README.md), 백엔드 설치와 설정은 [백엔드 README](../backend/README.md)를 참고하세요.

## 실행

프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

프런트엔드는 `http://localhost:5173`에서 실행됩니다. 사건 목록과 상태 변경을 사용하려면 백엔드도 `http://localhost:8000`에서 실행해야 합니다.

## 주요 파일

- `src/App.jsx`: 인증, 사건, 발주, 질문 화면
- `src/api/client.js`: 공통 HTTP 요청과 오류 처리
- `src/api/auth.js`: 회원가입, 로그인, 로그아웃, 현재 사용자 API
- `src/api/camera.js`: 로그인 세션으로 실시간 탐지 화면 조회
- `src/api/events.js`: 사건 API 호출 함수
- `src/api/demand.js`: 발주 API 호출 함수
- `src/api/questions.js`: 점주 질문 및 질문 로그 API 호출
- `src/main.jsx`: React 애플리케이션 진입점

## 검증

```powershell
npm run lint
npm run build
```

## 카메라 화면과 API 프록시

Vite 개발 서버와 Nginx는 `/api`만 백엔드로 프록시합니다. 점주 브라우저는 탐지 PC 포트에 직접 연결하지 않습니다. 관제 화면은 로그인 세션으로 `/api/camera/live-frame?after={seq}`를 롱폴링하고, 백엔드가 내부 탐지 토큰을 사용해 탐지 PC의 새 JPEG 화면을 중계합니다. 개발 백엔드 주소가 다르면 Vite 실행 전에 `VITE_API_PROXY_TARGET`을 지정하세요.

```powershell
$env:VITE_API_PROXY_TARGET = "http://localhost:8000"
npm run dev -- --host 0.0.0.0
```

보호 API에는 로그인 응답의 `session_id`가 `X-Session-ID` 헤더로 전달됩니다. 관제 화면은 최근 사건 5건만 요청하며, 사건 탭은 전체 이력을 조회합니다. USB 웹캠은 브라우저가 아니라 탐지 PC에 연결합니다.

## 질문 기능

질문 화면은 `/api/ask`로 요청하고, `/api/ask/{question_session_id}/log`에서 실행 도구와 조회 로그를 가져옵니다. 이 기능을 사용하려면 백엔드뿐 아니라 `qna-agent`, `qna-llm`과 필요한 모델·규정 데이터도 준비되어야 합니다. 수요·질문 서비스와 전체 실행 방법은 [프로젝트 README](../README.md)를 참고하세요.

## 기여

변경 후 `npm run lint`와 `npm run build`를 실행하고, 문서와 설명은 프로젝트 지침에 따라 한국어로 작성합니다.

## 라이선스

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
