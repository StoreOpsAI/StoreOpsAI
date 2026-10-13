"""배포 폴더의 lying_core가 학습·평가 코드(storeops_ai_fr_evt)와 같은 값을 내는지 검사한다(개발용, 팀원 전달 시 빼도 된다).

실행: storeops_ai_fr_evt에서 `venv/Scripts/python.exe ../analysis/check_parity.py`
1) pose_features: 학습에 쓴 관절 캐시의 사람 수백 명에서 두 구현의 특징이 같다
2) LyingMLP: 학습 코드의 MLP 점수와 같다
3) LyingTracker: 같은 점수·박스 열에서 models/fall_detector.lying_alarms와 같은 경보를 낸다
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "storeops_ai_fr_evt"))

import lying_core  # noqa: E402
from models.fall_detector import lying_alarms  # noqa: E402
from pipelines.lying_manifest import MANIFEST_PATH  # noqa: E402
from pipelines.lying_models import LyingMLP as TrainMLP  # noqa: E402
from pipelines.lying_pose_dataset import load_clip, pose_features  # noqa: E402
import json  # noqa: E402

HOLD = json.loads((HERE / "config.json").read_text(encoding="utf-8"))["hold_sec"]  # 배포 설정의 유지 시간(학습 쪽 LYING_HOLD_SEC와 같아야 한다)

clips = [c for c in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["train"] if c.get("first_pass")][:5]
deploy, train = lying_core.LyingMLP(HERE / "weights" / "lying_mlp.pt"), TrainMLP.load(HERE / "weights" / "lying_mlp.pt")
npz_dir = HERE.parent / "storeops_ai_fr_evt" / "output" / "lying_pose"
for clip in clips:
    data = np.load(npz_dir / f"{clip['clip']}.npz")
    a = np.array([lying_core.pose_features(b, k) for b, k in zip(data["box"], data["kpt"])])
    b = np.array([pose_features(bx, k) for bx, k in zip(data["box"], data["kpt"])])
    assert np.allclose(a, b), "pose_features 불일치"
    s1, s2 = deploy(a), train(b)
    assert np.allclose(s1, s2, atol=1e-5), "MLP 점수 불일치"
    by_frame = {}
    for f, box, s in zip(data["frame"], data["box"], s1):
        if s >= 0.5:
            by_frame.setdefault(int(f), []).append(tuple(map(float, box)))
    samples = [(f, by_frame.get(f, [])) for f in range(int(data["n_frames"]))]
    expected = lying_alarms(samples, 3.0)
    tracker, got = lying_core.LyingTracker(3.0, HOLD, 0.3, 1), []
    for f, boxes in samples:
        got += tracker.update(f, boxes)
    assert got == expected, f"경보 불일치 {clip['clip']}: {got} != {expected}"
    print(clip["clip"][-30:], "사람", len(a), "경보", len(got), "일치")
print("ok")
