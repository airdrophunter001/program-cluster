"""
3D-aware 경로 손실 모델
Building3D + 3D 좌표(x,y,z)를 사용해 층별 신호 맵 계산
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wifi_heatmap.core.propagation import (
    PropagationParams,
    compute_path_loss,
    received_power_dbm,
    compute_sinr_db,
    shannon_capacity_mbps,
)
from wifi_heatmap.core.building_3d import Building3D, Floor
from wifi_heatmap.core.router import Router, check_channel_interference

NOISE_FLOOR_DBM = -100.0
RX_GAIN_DBI = 0.0


@dataclass
class FloorSignalMap:
    """한 층의 신호 강도 맵"""
    floor_number: int
    floor_label: str
    xs: np.ndarray
    ys: np.ndarray
    rssi_grid: np.ndarray    # dBm
    sinr_grid: np.ndarray    # dB
    capacity_grid: np.ndarray  # Mbps

    @property
    def coverage_pct(self) -> float:
        return 100.0 * (self.rssi_grid >= -75.0).sum() / self.rssi_grid.size

    def coverage_at(self, threshold_dbm: float = -75.0) -> float:
        return 100.0 * (self.rssi_grid >= threshold_dbm).sum() / self.rssi_grid.size


class PathLossModel3D:
    """
    3D 건물 + 다중 AP → 층별 신호/SINR/용량 맵 계산기

    AP는 router.height_m 속성으로 설치 높이를 지정.
    층 번호 지정 없이 height_m 기준으로 자동으로 어느 층인지 판별.
    """

    def __init__(
        self,
        building: Building3D,
        params: PropagationParams,
        noise_floor_dbm: float = NOISE_FLOOR_DBM,
    ):
        self.building = building
        self.params = params
        self.noise_floor_dbm = noise_floor_dbm

    # ── 단일 AP × 단일 층 ────────────────────────────────────────────
    def _rssi_at_point(
        self,
        router: Router,
        rx_x: float,
        rx_y: float,
        rx_z: float,
    ) -> float:
        """3D 거리 + 벽/층간 감쇠를 반영한 수신 전력"""
        tx = (router.x, router.y, router.height_m)
        rx = (rx_x, rx_y, rx_z)

        dist3d = math.sqrt(
            (tx[0] - rx[0]) ** 2 +
            (tx[1] - rx[1]) ** 2 +
            (tx[2] - rx[2]) ** 2
        )
        if dist3d < 0.1:
            dist3d = 0.1

        pl = compute_path_loss(dist3d, router.frequency_hz, self.params)
        wall_loss = self.building.wall_attenuation_3d(tx, rx)

        return received_power_dbm(
            tx_power_dbm=router.tx_power_dbm,
            tx_gain_dbi=router.antenna_gain_dbi,
            rx_gain_dbi=RX_GAIN_DBI,
            path_loss_db=pl,
            wall_loss_db=wall_loss,
        )

    def compute_floor_maps(
        self,
        routers: list[Router],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> list[FloorSignalMap]:
        """모든 층에 대해 신호/SINR/용량 맵 계산"""
        active = [r for r in routers if r.enabled]
        results: list[FloorSignalMap] = []

        for floor in self.building.floors:
            rx_z = floor.z_mid  # 수신 높이 = 층 중간 (약 1.4m)
            rows, cols = len(ys), len(xs)

            # AP별 RSSI 그리드 계산
            per_ap: list[np.ndarray] = []
            for router in active:
                grid = np.zeros((rows, cols), dtype=np.float32)
                for j, y in enumerate(ys):
                    for i, x in enumerate(xs):
                        grid[j, i] = max(
                            self._rssi_at_point(router, x, y, rx_z),
                            self.noise_floor_dbm,
                        )
                per_ap.append(grid)

            if not per_ap:
                empty = np.full((rows, cols), self.noise_floor_dbm, dtype=np.float32)
                results.append(FloorSignalMap(
                    floor_number=floor.number, floor_label=floor.label,
                    xs=xs, ys=ys,
                    rssi_grid=empty,
                    sinr_grid=np.zeros((rows, cols), dtype=np.float32),
                    capacity_grid=np.zeros((rows, cols), dtype=np.float32),
                ))
                continue

            all_stacked = np.stack(per_ap, axis=0)
            best_rssi = np.max(all_stacked, axis=0)
            best_idx  = np.argmax(all_stacked, axis=0)

            sinr_grid     = np.zeros((rows, cols), dtype=np.float32)
            capacity_grid = np.zeros((rows, cols), dtype=np.float32)

            for j in range(rows):
                for i in range(cols):
                    si = int(best_idx[j, i])
                    sig = all_stacked[si, j, i]
                    intf = [
                        all_stacked[k, j, i]
                        for k in range(len(active))
                        if k != si and check_channel_interference(active[si], active[k]) > 0
                    ]
                    sinr = compute_sinr_db(sig, intf, self.noise_floor_dbm)
                    sinr_grid[j, i] = sinr
                    bw = active[si].channel_bandwidth_mhz
                    capacity_grid[j, i] = shannon_capacity_mbps(bw, sinr)

            results.append(FloorSignalMap(
                floor_number=floor.number, floor_label=floor.label,
                xs=xs, ys=ys,
                rssi_grid=best_rssi,
                sinr_grid=sinr_grid,
                capacity_grid=capacity_grid,
            ))

        return results

    def coverage_summary(
        self,
        floor_maps: list[FloorSignalMap],
        threshold_dbm: float = -75.0,
    ) -> dict:
        """전체 건물 커버리지 요약"""
        all_rssi = np.concatenate([fm.rssi_grid.flatten() for fm in floor_maps])
        all_sinr = np.concatenate([fm.sinr_grid.flatten() for fm in floor_maps])
        total = all_rssi.size
        covered = (all_rssi >= threshold_dbm).sum()

        per_floor = {
            fm.floor_label: {
                "coverage_pct": fm.coverage_at(threshold_dbm),
                "mean_rssi": float(np.mean(fm.rssi_grid[fm.rssi_grid > self.noise_floor_dbm])),
            }
            for fm in floor_maps
        }

        return {
            "total_coverage_pct": 100.0 * covered / total,
            "mean_rssi_dbm": float(np.mean(all_rssi[all_rssi > self.noise_floor_dbm])),
            "mean_sinr_db": float(np.clip(np.mean(all_sinr), 0, 60)),
            "per_floor": per_floor,
            "num_floors": len(floor_maps),
        }
