"""외부 데이터 없이 연구 원본의 특징·모델 정의를 서비스와 대조한다.

연구 모듈의 최상위 import는 로컬 데이터 설정까지 불러오므로 AST로 필요한
정의만 읽는다. 원본 함수/클래스 본문을 그대로 실행하며 학습 결과를 대체하지 않는다.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import sys

import numpy as np
import pytest
import torch
from torch import nn

from models.pose_head import PoseHead, frame_feature, window_feature, pick_largest
from utils import relation_features as service_rel

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'video_research/storeops_ai_fr_evt/pipelines'


def definitions(path, names, **globals_):
    """선택한 연구 정의를 원문 그대로 격리 실행한다."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    env = dict(np=np, torch=torch, nn=nn, **globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def test_pose_features_and_selection():
    """학습과 추론의 53채널 수치, 빈 프레임, 최대 인물 선택을 확인한다."""
    research = definitions(RESEARCH / 'pose_head.py', ['frame_feature'])
    rng = np.random.default_rng(42)
    for _ in range(50):
        kpt = rng.uniform(0, 100, (17, 3)).astype(np.float32)
        box = np.array([2, 3, 80, 160], np.float32)
        np.testing.assert_array_equal(frame_feature(kpt, box), research['frame_feature'](kpt, box))
    large = (kpt, box)
    assert pick_largest([(kpt, [0, 0, 1, 1]), large]) is large
    x = window_feature([None] * 11 + [large])
    assert x.shape == (12, 53)
    assert not x[:11].any()


@pytest.mark.parametrize('dim', [53, 73])
def test_gru_checkpoint_contract(tmp_path, dim):
    """연구 구조로 만든 합성 가중치가 서비스에 strict 로드되고 같은 로짓을 낸다.

73차원은 연구 클래스의 DIM만 주입한 구조 검사이며 미제공 P+R 학습 코드 검증은 아니다.
"""
    env = definitions(RESEARCH / 'pose_head.py', ['PoseGRU'], DIM=dim, NC=6)
    torch.manual_seed(42)
    model = env['PoseGRU']().eval()
    path = tmp_path / 'synthetic.pt'
    torch.save({'model': model.state_dict()}, path)
    service = PoseHead(path)
    x = np.random.default_rng(42).normal(size=(12, dim)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x[None])).numpy()[0]
    np.testing.assert_allclose(service.logits(x), expected, rtol=1e-6, atol=1e-6)
    assert service.uses_relation == (dim == 73)
    with pytest.raises(ValueError):
        service.logits(np.zeros((11, dim), np.float32))


def test_relation_scene_channels():
    """사람·물건 등장/이동/소실의 14채널을 연구 원본과 대조한다."""
    env = definitions(RESEARCH / 'relation_windows.py', ['centers', 'point_box_distance', 'frame_channels'],
                      CHANNELS=service_rel.CHANNELS, FAR=1.0, CLIP=5.0)
    prev_r = prev_s = None
    for boxes, objects in [([[0, 0, 40, 100]], [[40, 40, 60, 60]]),
                            ([[10, 0, 50, 100]], [[50, 40, 70, 60]]),
                            ([], [[50, 40, 70, 60]]), ([], [])]:
        pb = np.array(boxes, np.float32).reshape(-1, 4)
        ob = np.array(objects, np.float32).reshape(-1, 4)
        oc = np.full(len(ob), .8, np.float32)
        a, prev_r = env['frame_channels'](pb, ob, oc, prev_r)
        b, prev_s = service_rel.frame_channels(pb, ob, oc, prev_s)
        np.testing.assert_allclose(a, b)
    service_rel._self_check()  # 손목 6채널은 서비스 내 경계 조건만 확인


def test_yolo_tracking_interface():
    """실제 YOLO 가중치 없이 전달 인자와 결과 인터페이스만 확인한다."""
    calls = []
    result = SimpleNamespace(keypoints=None, boxes=None)
    class FakeYOLO:
        def __init__(self, path):
            self.path = path
        def track(self, frame, **kwargs):
            calls.append(kwargs)
            return [result]
    spec = importlib.util.spec_from_file_location('parity_yolo', ROOT / 'storeops_ai/models/yolo_detector.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=FakeYOLO)}):
        spec.loader.exec_module(module)
    detector = module.YOLODetector('yolo11s-pose.pt', 'botsort_storeops.yaml', .6, 0, 1280)
    assert detector.track(np.zeros((8, 8, 3), np.uint8)) is result
    assert calls == [dict(tracker='botsort_storeops.yaml', classes=[0], conf=.6, persist=True, verbose=False, imgsz=1280)]
    assert detector.keypoints_by_track(result) == {}
