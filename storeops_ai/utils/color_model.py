"""빈 화면 통계 색 모델: 픽셀마다 밝기 왜곡·색 왜곡을 정규화하고, 빈 화면 분포에서 문턱을 자동으로 고른다.

방법: Horprasert, Harwood, Davis, "A Statistical Approach for Real-time Robust Background Subtraction and Shadow Detection",
ICCV'99 FRAME-RATE Workshop (본문 읽음, papers_shadow_light/Horprasert1999.pdf).
- 식 3~8: 빈 화면 N장의 픽셀별 평균 mu, 표준편차 sigma, 밝기 왜곡 변동 a = RMS(alpha-1), 색 왜곡 변동 b = RMS(CD).
- 식 9~10: alpha_hat = (alpha-1)/a, cd_hat = CD/b. §4.3: 빈 화면 전체 값의 히스토그램에서 검출률 r 분위로 문턱.
- 식 12: 물건(F) = cd_hat > tau_cd 또는 너무 어두움, 배경 = tau_a2 < alpha_hat < tau_a1, 그림자 = alpha_hat < 0, 나머지 = 빛 얼룩.
- §5: b가 아주 작은 픽셀에 오검출이 몰리므로 b 최솟값을 두고, 두 픽셀 무리의 오류율이 같아지는 값을 찾는다.
논문과 다른 점(행동인식_분석.md 11-추가-6):
- 너무 어두움 하한(논문 tau_alpha_lo=0.4)은 alpha_hat이 아니라 alpha에 적용한다. 본문은 alpha_hat 하한이라 쓰지만 0.4는 alpha에서만 뜻이
  통한다(alpha_hat 하한 0.4면 배경 대부분이 물건이 된다). 근거 약함.
- sigma=0 픽셀의 최솟값은 논문에 수치가 없어 8비트 양자화 잡음 1/sqrt(12)를 쓴다. 근거 약함.
- b 최솟값 탐색은 별도 시험 영상 대신 보정 프레임을 앞뒤 절반으로 나눠 앞쪽으로 모델을, 뒤쪽으로 오류율을 잰다.
- §3 clipping: 잘린(포화) 화소는 분포가 한 점에 몰려 왜곡을 잴 수 없다. 논문은 s_i 최솟값만 두지만, 그래도 sigma로 나눈 공간이 크게 늘어나
  창문·햇빛 자국이 물건으로 잡혔다(시연 run27 E004). 그래서 보정 때 잘린 화소(COLOR_CLIP_LEVEL에 닿았거나 N장 내내 값이 같음)와 지금 그 값 이상인 화소는
  물건도 배경도 아닌 '잴 수 없음'으로 둔다. 대가: 포화된 자리(창문 유리, 햇빛 자국)와 잘림 값으로 날아간 흰 물건은 못 잡는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config.config import COLOR_CLIP_LEVEL as CEILING   # 잘림 판정 값(카메라 측정값)

SIGMA_MIN = 1.0 / np.sqrt(12.0)   # 8비트 양자화 잡음의 표준편차(근거 약함, 위 설명)


@dataclass
class ColorModel:
    mu: np.ndarray       # (h, w, 3) 빈 화면 평균(BGR)
    sigma: np.ndarray    # (h, w, 3)
    a: np.ndarray        # (h, w)
    b: np.ndarray        # (h, w) b 최솟값 적용 후
    tau_cd: float
    tau_a1: float
    tau_a2: float
    dark_min: float      # alpha가 이보다 작으면 그림자가 아니라 물건(논문 tau_alpha_lo)
    b_min: float
    clip: np.ndarray | None = None   # (h, w) 보정 때 잘린 화소(§3 clipping). 잴 수 없어 판정하지 않는다

    def classify(self, img: np.ndarray, expected: np.ndarray | None = None, noise_gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(물건, 배경) 마스크. 그림자·빛 얼룩은 둘 다 아니다. expected를 주면 mu 대신 그 화면과 비교한다(잡음은 noise_gain배)."""
        mu = self.mu if expected is None else expected.astype(np.float32)
        alpha, cd = distortions(img, mu, self.sigma * noise_gain)
        ah, ch = (alpha - 1.0) / self.a, cd / (self.b * noise_gain)   # 이 a, b는 빈 화면 기준이라 noise_gain만큼 키운 cd를 같은 비율로 되돌린다
        unmeasurable = (img >= CEILING).any(-1) if self.clip is None else self.clip | (img >= CEILING).any(-1)
        fg = ((ch > self.tau_cd) | (alpha < self.dark_min)) & ~unmeasurable
        bg = ~fg & ~unmeasurable & (ah < self.tau_a1) & (ah > self.tau_a2)
        return fg, bg

    def background_image(self) -> np.ndarray:
        return np.clip(self.mu, 0, 255).astype(np.uint8)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mu=self.mu, sigma=self.sigma, a=self.a, b=self.b, clip=self.clip if self.clip is not None else np.zeros(self.a.shape, bool),
                            scalars=np.array([self.tau_cd, self.tau_a1, self.tau_a2, self.dark_min, self.b_min], np.float64))

    @staticmethod
    def load(path: Path) -> ColorModel | None:
        path = Path(path)
        if not path.is_file():
            return None
        z = np.load(path)
        return ColorModel(z["mu"], z["sigma"], z["a"], z["b"], *(float(x) for x in z["scalars"]), clip=z["clip"] if "clip" in z else None)   # 예전 파일은 다시 보정해야 clip이 생긴다


def distortions(img: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """식 5, 6: 채널을 sigma로 나눈 공간에서 밝기 왜곡 alpha와 색 왜곡 CD."""
    i, m = img.astype(np.float32) / sigma, mu / sigma
    alpha = (i * m).sum(-1) / np.maximum((m * m).sum(-1), 1e-12)
    cd = np.sqrt(((i - alpha[..., None] * m) ** 2).sum(-1))
    return alpha, cd


def _stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = frames.mean(0, dtype=np.float64).astype(np.float32)
    sigma = np.maximum(frames.std(0, dtype=np.float64).astype(np.float32), SIGMA_MIN)
    alpha, cd = zip(*(distortions(f, mu, sigma) for f in frames))
    alpha, cd = np.stack(alpha), np.stack(cd)
    a = np.maximum(np.sqrt(((alpha - 1.0) ** 2).mean(0)), 1e-6)   # 식 7. 0이면 나눗셈만 막는다(포화 픽셀)
    b = np.maximum(np.sqrt((cd ** 2).mean(0)), 1e-6)              # 식 8. 같은 이유
    return mu, sigma, a, b, alpha, cd


def _find_b_min(train: np.ndarray, test: np.ndarray, r: float, dark_min: float) -> float:
    """§5: b 최솟값 후보마다 오검출률을 b가 작은 무리와 큰 무리로 나눠 재고, 둘이 같아지는 값을 고른다."""
    mu, sigma, _, b, _, cd_tr = _stats(train)
    alpha_te, cd_te = zip(*(distortions(f, mu, sigma) for f in test))
    alpha_te, cd_te = np.stack(alpha_te), np.stack(cd_te)
    dark = (alpha_te < dark_min).mean(0)
    best, best_gap = float(np.quantile(b, 0.5)), np.inf
    for cand in np.quantile(b, np.linspace(0.05, 0.95, 19)):   # ponytail: 분위 19개 격자 탐색, 교차점이 격자 사이면 가장 가까운 후보
        bb = np.maximum(b, cand)
        tau = np.quantile(cd_tr / bb, r)
        err = np.maximum((cd_te / bb > tau).mean(0), dark)
        small = b <= cand
        if small.all() or not small.any():
            continue
        gap = abs(err[small].mean() - err[~small].mean())
        if gap < best_gap:
            best, best_gap = float(cand), gap
    return best


def fit(frames: Sequence[np.ndarray], r: float, dark_min: float) -> ColorModel:
    """빈 화면 프레임들(같은 크기 BGR)로 모델과 문턱을 정한다. 사람이 손으로 넣는 값은 없다(r, dark_min은 논문값)."""
    f = np.stack(frames).astype(np.float32)
    if len(f) < 4:
        raise ValueError(f"빈 화면 프레임이 너무 적습니다({len(f)}장, 최소 4장)")
    b_min = _find_b_min(f[: len(f) // 2], f[len(f) // 2:], r, dark_min)
    mu, sigma, a, b, alpha, cd = _stats(f)
    b = np.maximum(b, b_min)
    clip = (f >= CEILING).any(0).any(-1) | (f.std(0) == 0).all(-1)   # §3: 잘림 값에 닿았거나 값이 한 점에 몰린(분산 0) 화소
    if clip.all():
        raise ValueError("빈 화면의 모든 화소가 잘렸거나 값이 그대로입니다(같은 장 반복·노출 과다). 다시 보정하세요")
    ah, ch = (alpha - 1.0)[:, ~clip] / a[~clip], cd[:, ~clip] / b[~clip]   # 문턱은 잴 수 있는 화소로만 정한다
    return ColorModel(mu, sigma, a, b, float(np.quantile(ch, r)), float(np.quantile(ah, r)), float(np.quantile(ah, 1.0 - r)),
                      dark_min, b_min, clip)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (60, 80, 3)).astype(np.float32)
    shot = lambda img: np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)   # 카메라 잡음
    m = fit([shot(base) for _ in range(100)], 0.9999, 0.4)
    fg, bg = m.classify(shot(base))
    assert fg.mean() < 0.005 and bg.mean() > 0.98, (fg.mean(), bg.mean())                  # 빈 화면은 거의 배경
    obj = base.copy(); obj[20:40, 30:50] = (40, 200, 60)                                     # 색이 다른 물건
    assert m.classify(shot(obj))[0][22:38, 32:48].mean() > 0.95
    for k in (0.6, 1.3, 2.0):                                                                # 그림자, 빛 얼룩(상한 없음)
        fg, bg = m.classify(shot(base * k))
        ok = (base * k + 8 < 255).all(-1)                                                    # 255 근처는 잡음이 잘려 아래 따로 본다
        assert fg[ok].mean() < 0.01 and bg[ok].mean() < 0.05, (k, fg[ok].mean())             # 물건도 배경도 아니다
    assert not m.classify(np.full_like(shot(base), 255))[0].any()                            # 지금 255로 잘린 화소는 잴 수 없다(§3)
    sat = base.copy(); sat[:10] = 255
    ms = fit([shot(sat) for _ in range(100)], 0.9999, 0.4)
    assert ms.clip[:10].all() and not ms.classify(shot(sat * 0.8))[0][:10].any()          # 보정 때 잘린 창문이 어두워져도 물건이 아니다
    assert ms.classify(shot(obj))[0][22:38, 32:48].mean() > 0.95                             # 나머지 자리는 그대로 잡는다
    assert m.classify(shot(base * 0.3))[0].mean() > 0.95                                     # 너무 어두우면 물건(dark_min)
    fg, _ = m.classify(shot(base), expected=shot(base), noise_gain=np.sqrt(2.0))             # 두 장 비교(움직임 판정)
    assert fg.mean() < 0.01, fg.mean()
    print("ok", round(m.tau_cd, 2), round(m.tau_a1, 2), round(m.tau_a2, 2), round(m.b_min, 3))


if __name__ == "__main__":
    _self_check()
