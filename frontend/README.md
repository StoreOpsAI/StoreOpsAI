<p align="center">
	<img src="https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white" alt="React 19">
	<img src="https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white" alt="Vite 8">
</p>

# StoreOps AI 프런트엔드

StoreOps AI 점주용 웹 애플리케이션입니다. React 19와 Vite를 사용하며, 로그인한 점주가 매장 사건과 상세 정보, 처리 이력을 확인하고 상태를 변경할 수 있습니다.

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

- `src/App.jsx`: 사건 목록, 상세 정보, 상태 변경 화면
- `src/api/client.js`: 공통 HTTP 요청과 오류 처리
- `src/api/events.js`: 사건 API 호출 함수
- `src/main.jsx`: React 애플리케이션 진입점

## 검증

```powershell
npm run lint
npm run build
```

## 앱 안내

StoreOps AI 점주용 화면입니다. 로그인한 점주가 매장 사건과 상세 정보, 처리 이력을 확인하고 상태를 변경할 수 있습니다.

개발 서버는 `/api` 요청을 `http://localhost:8000`으로, `/detector` 요청을 `http://localhost:8100`으로 프록시합니다. 주소를 바꾸려면 Vite 실행 전에 `VITE_API_PROXY_TARGET` 또는 `VITE_DETECTOR_PROXY_TARGET` 환경변수를 지정하세요.

```powershell
$env:VITE_API_PROXY_TARGET = "http://localhost:8000"
$env:VITE_DETECTOR_PROXY_TARGET = "http://localhost:8100"
npm run dev -- --host 0.0.0.0
```

로그인 후 보호 API에는 `X-Session-ID` 헤더를 사용합니다. 새 매장에는 사건이 자동으로 생성되지 않으므로 CCTV 탐지 서비스에서 사건을 수신해야 목록에 데이터가 표시됩니다. 하위 서비스 설정은 [프로젝트 README](../README.md)를 참고하세요.

| 파일                | 설명                            |
| ------------------- | ------------------------------- |
| `src/App.jsx`       | 인증, 사건 및 발주 화면         |
| `src/api/client.js` | 공통 요청, 세션 헤더, 오류 처리 |
| `src/api/auth.js`   | 인증 API 호출                   |
| `src/api/events.js` | 사건 및 미디어 API 호출         |
| `vite.config.js`    | 개발 서버와 API 프록시 설정     |

## 기여

변경 후 `npm run lint`와 `npm run build`를 실행하고, 문서와 설명은 프로젝트 지침에 따라 한국어로 작성합니다.

## 라이선스

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
