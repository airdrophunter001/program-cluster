"""
빅데이터 기반 최소 AP 배치 탐색
- AP 개수(1..max_routers)별로 다수의 무작위 배치 후보를 Monte-Carlo로
  시뮬레이션하여 커버리지/SINR 통계(평균/표준편차/분포)를 수집한다.
- 목표 커버리지(target_coverage_pct)를 만족하는 최소 AP 개수와
  그 중 최고 성능 배치를 추천한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from wifi_heatmap.core.building_3d import Building3D
from wifi_heatmap.core.propagation import PropagationParams
from wifi_heatmap.core.router import Router
from wifi_heatmap.models.path_loss_model_3d import PathLossModel3D


@dataclass
class SampleResult:
    routers: list[Router]
    coverage_pct: float
    mean_rssi: float
    mean_sinr: float


@dataclass
class CountStats:
    num_aps: int
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def coverage_values(self) -> np.ndarray:
        return np.array([s.coverage_pct for s in self.samples])

    @property
    def best(self) -> SampleResult:
        return max(self.samples, key=lambda s: s.coverage_pct)

    def stats(self) -> dict:
        cov = self.coverage_values
        return {
            "num_aps": self.num_aps,
            "n_samples": len(self.samples),
            "mean": float(np.mean(cov)),
            "std": float(np.std(cov)),
            "min": float(np.min(cov)),
            "max": float(np.max(cov)),
            "median": float(np.median(cov)),
            "p90": float(np.percentile(cov, 90)),
            "best_mean_rssi": self.best.mean_rssi,
            "best_mean_sinr": self.best.mean_sinr,
        }


@dataclass
class SearchResult:
    by_count: dict[int, CountStats]
    recommended_num_aps: Optional[int]
    threshold_dbm: float
    target_coverage_pct: float
    total_samples: int
    total_data_points: int


def _random_router(building: Building3D, band: str) -> Router:
    floor = random.choice(building.floors)
    x = random.uniform(0.5, building.footprint_width - 0.5)
    y = random.uniform(0.5, building.footprint_height - 0.5)
    channels = [1, 6, 11] if band == "2.4GHz" else list(range(36, 165, 4))
    ch = random.choice(channels)
    tx = random.uniform(15, 23)
    inner_h = random.uniform(1.0, max(floor.height - 0.5, 1.0))
    return Router(
        x=x, y=y, band=band,
        tx_power_dbm=tx, channel=ch,
        height_m=floor.z_bottom + inner_h,
    )


def run_big_data_search(
    building: Building3D,
    params: PropagationParams,
    band: str,
    xs: np.ndarray,
    ys: np.ndarray,
    threshold_dbm: float = -75.0,
    target_coverage_pct: float = 90.0,
    max_routers: int = 4,
    samples_per_count: int = 20,
    noise_floor_dbm: float = -100.0,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SearchResult:
    """AP 1개~max_routers개 각각에 대해 samples_per_count개의 무작위 배치를
    시뮬레이션하고, 목표 커버리지를 처음 만족하는 최소 AP 개수를 찾는다."""
    if seed is not None:
        random.seed(seed)

    model = PathLossModel3D(building, params, noise_floor_dbm=noise_floor_dbm)

    by_count: dict[int, CountStats] = {}
    recommended: Optional[int] = None
    total_steps = max_routers * samples_per_count
    step = 0
    cells_per_eval = len(xs) * len(ys) * building.num_floors

    for n in range(1, max_routers + 1):
        cstats = CountStats(num_aps=n)
        for _ in range(samples_per_count):
            routers = [_random_router(building, band) for _ in range(n)]
            floor_maps = model.compute_floor_maps(routers, xs, ys)
            summary = model.coverage_summary(floor_maps, threshold_dbm)
            cstats.samples.append(SampleResult(
                routers=routers,
                coverage_pct=summary["total_coverage_pct"],
                mean_rssi=summary["mean_rssi_dbm"],
                mean_sinr=summary["mean_sinr_db"],
            ))
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)

        by_count[n] = cstats
        if recommended is None and cstats.best.coverage_pct >= target_coverage_pct:
            recommended = n

    return SearchResult(
        by_count=by_count,
        recommended_num_aps=recommended,
        threshold_dbm=threshold_dbm,
        target_coverage_pct=target_coverage_pct,
        total_samples=total_steps,
        total_data_points=total_steps * cells_per_eval,
    )
