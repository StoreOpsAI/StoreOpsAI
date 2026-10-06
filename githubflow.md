# GitHub Flow

이 저장소는 `main` 브랜치와 기능별 작업 브랜치를 사용합니다. `main`에 직접 커밋하거나 푸시하지 않고, 모든 변경은 Pull Request(PR)로 반영합니다.

## 작업 시작

최신 `main`을 가져온 뒤 작업 목적을 알아보기 쉬운 브랜치를 만듭니다.

```bash
git switch main
git pull origin main
git switch -c feature/기능이름
```

버그 수정 브랜치는 `fix/문제이름`, 문서 작업 브랜치는 `docs/문서이름`처럼 목적에 맞게 이름을 정합니다.

## 변경 및 커밋

작업을 마치면 변경 내용을 검토하고, 관련 검증을 실행한 다음 변경 사항을 커밋합니다.

```bash
git status
git diff
git add 변경할-파일
git commit -m "태그: 변경 내용 요약"
```

커밋 메시지는 한국어로 작성하고 다음 태그를 사용합니다.

| 태그       | 사용 기준                |
| ---------- | ------------------------ |
| `feat`     | 기능 추가                |
| `fix`      | 버그 수정                |
| `docs`     | 문서 수정                |
| `style`    | 코드 형식 변경           |
| `refactor` | 동작 변경 없는 구조 개선 |
| `test`     | 테스트 추가 또는 수정    |
| `chore`    | 빌드·도구·환경 설정 변경 |

## 푸시 및 Pull Request

작업 브랜치를 원격 저장소에 올리고 `main`을 대상으로 PR을 생성합니다.

```bash
git push -u origin feature/기능이름
```

PR에는 변경 목적과 주요 내용을 적고, 관련 테스트 및 검증 결과를 기록합니다. 리뷰 요청을 보내고, 리뷰 의견을 반영한 뒤 최소 한 명의 승인을 받아 병합합니다. `main`에 직접 푸시하지 않습니다.

## 병합 후

PR이 병합되면 로컬 `main`을 최신 상태로 갱신하고, 병합된 작업 브랜치를 정리합니다.

```bash
git switch main
git pull origin main
git branch -d feature/기능이름
```
