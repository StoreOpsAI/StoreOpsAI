# 🏪 StoreOpsAI

> StoreOpsAI 팀 프로젝트 저장소입니다.  
> 효율적인 협업을 위해 아래의 **GitHub Flow 전략**과 **커밋 컨벤션**을 반드시 준수해 주세요!

---

## 🚀 1. 개발 프로세스 (GitHub Flow)

우리 팀은 **`main` 브랜치**와 기능별 **`feature/기능이름` 브랜치**만 사용합니다.  
`main` 브랜치는 언제나 배포 가능한 상태를 유지해야 하므로 직접 푸시(Push)가 불가능하며, 무조건 **Pull Request(PR)**를 거쳐야 합니다.

### 💻 일상 작업 순서 (Git 명령어)

1. **최신 코드 가져오기 (작업 시작 전 필수)**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **새로운 기능 브랜치 생성 및 이동**
   ```bash
   # 브랜치명 예시: feature/login, fix/signup-bug
   git checkout -b feature/기능이름
   ```

3. **코드 수정 후 커밋하기**
   ```bash
   git add .
   git commit -m "태그: 작업 내용 한 줄 요약"
   ```

4. **원격 저장소(GitHub)에 브랜치 푸시**
   ```bash
   git push origin feature/기능이름
   ```

5. **Pull Request(PR) 생성 및 머지(Merge)**
   - GitHub 웹사이트에 접속하여 `main` 브랜치 방향으로 PR을 생성합니다.
   - **최소 1명 이상의 팀원에게 코드 리뷰와 승인(Approve)**을 받은 후 `Merge`합니다.

---

## 📝 2. 커밋 컨벤션 (Commit Convention)

작성한 커밋 메시지만 보고도 변경 사항을 파악할 수 있도록 아래 규칙을 따릅니다.

### 📌 커밋 메시지 구조
```text
태그: 한 줄 요약

- 필요한 경우 자세한 설명 추가 (본문)
```

### 🏷️ 주요 태그 종류

| 태그 (Tag) | 설명 | 예시 |
| :--- | :--- | :--- |
| **`feat`** | 새로운 기능 추가 | `feat: 카카오 소셜 로그인 구현` |
| **`fix`** | 버그 및 오류 수정 | `fix: 이메일 중복 체크 에러 수정` |
| **`docs`** | 문서 수정 (README, 주석 등) | `docs: README 개발 환경 안내 추가` |
| **`style`** | 코드 포맷팅 (세미콜론, 줄바꿈 등) | `style: 코드 인덴트 및 포맷 정렬` |
| **`refactor`**| 코드 리팩토링 (로직 변경 없는 구조 개선) | `refactor: 로그인 컴포넌트 구조 분리` |
| **`chore`** | 빌드 설정, 패키지 매니저, 환경 설정 변경 | `chore: 라이브러리 의존성 패키지 설치` |

---

## 🛠️ 3. 시작하기 (팀원 가이드)

프로젝트를 처음 다운로드하여 개발 환경을 세팅하는 방법입니다.

```bash
# 1. 저장소 클론
git clone https://github.com/StoreOpsAI/StoreOpsAI.git

# 2. 프로젝트 폴더 이동

