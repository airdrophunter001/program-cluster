"""
통합 경로 손실 모델: 전파 모델 + 환경 효과 결합
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wifi_heatmap.core.propagation import (
    PropagationModel,
    PropagationParams,
    compute_path_loss,
    received_power_dbm,
    compute_sinr_db,
    shannon_capacity_mbps,
)
from wifi_heatmap.core.environment import BuildingEnvironment, precompute_wall_attenuation_grid
from wifi_heatmap.core.router import Router, check_channel_interference


NOISE_FLOOR_DBM = -100.0
RX_GAIN_DBI = 0.0  # 수신 안테나 이득 (단말 기준)


@dataclass
class SignalMap:
    """단일 AP에 대한 신호 강도 맵"""
    router: Router
    band: str
    xs: np.ndarray
    ys: np.ndarray
    rssi_grid: np.ndarray       # dBm
    wall_loss_grid: np.ndarray  # dB
    sinr_grid: Optional[np.ndarray] = None  # dB
    capacity_grid: Optional[np.ndarray] = None  # Mbps

    @property
    def shape(self) -> tuple:
        return self.rssi_grid.shape

    def get_coverage_mask(self, threshold_dbm: float = -75.0) -> np.ndarray:
        return self.rssi_grid >= threshold_dbm

    def coverage_percentage(self, threshold_dbm: float = -75.0) -> float:
        mask = self.get_coverage_mask(threshold_dbm)
        return 100.0 * mask.sum() / mask.size


class PathLossModel:
    """전파 모델 + 환경 + AP 목록을 결합한 신호 맵 계산기"""

    def __init__(
        self,
        environment: BuildingEnvironment,
        params: PropagationParams,
        noise_floor_dbm: float = NOISE_FLOOR_DBM,
    ):
        self.environment = environment
        self.params = params
        self.noise_floor_dbm = noise_floor_dbm

    def compute_signal_map(
        self,
        router: Router,
        xs: np.ndarray,
        ys: np.ndarray,
        use_wall_attenuation: bool = True,
    ) -> SignalMap:
        """단일 AP에 대한 전체 그리드 신호 강도 계산"""
        rows, cols = len(ys), len(xs)
        rssi_grid = np.full((rows, cols), self.noise_floor_dbm, dtype=np.float32)
        wall_grid = np.zeros((rows, cols), dtype=np.float32)

        if not router.enabled:
            return SignalMap(
                router=router, band=router.band,
                xs=xs, ys=ys,
                rssi_grid=rssi_grid, wall_loss_grid=wall_grid,
            )

        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                dist = router.distance_to(x, y)
                if dist < 0.1:
                    dist = 0.1

                pl = compute_path_loss(dist, router.frequency_hz, self.params)

                wall_loss = 0.0
                if use_wall_attenuation:
                    wall_loss = self.environment.compute_wall_attenuation(
                        router.position, (x, y)
                    )
                wall_grid[j, i] = wall_loss

                rx = received_power_dbm(
                    tx_power_dbm=router.tx_power_dbm,
                    tx_gain_dbi=router.antenna_gain_dbi,
                    rx_gain_dbi=RX_GAIN_DBI,
                    path_loss_db=pl,
                    wall_loss_db=wall_loss,
                )
                rssi_grid[j, i] = max(rx, self.noise_floor_dbm)

        return SignalMap(
            router=router, band=router.band,
            xs=xs, ys=ys,
            rssi_grid=rssi_grid, wall_loss_grid=wall_grid,
        )

    def compute_combined_map(
        self,
        routers: list[Router],
        xs: np.ndarray,
        ys: np.ndarray,
        use_wall_attenuation: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        다중 AP 결합: 최강 신호 맵, SINR 맵, 이론 용량 맵 반환
        Returns: (best_rssi_grid, sinr_grid, capacity_grid)
        """
        signal_maps = [
            self.compute_signal_map(r, xs, ys, use_wall_attenuation)
            for r in routers if r.enabled
        ]

        if not signal_maps:
            empty = np.full((len(ys), len(xs)), self.noise_floor_dbm, dtype=np.float32)
            return empty, empty, np.zeros_like(empty)

        all_grids = np.stack([sm.rssi_grid for sm in signal_maps], axis=0)
        best_rssi = np.max(all_grids, axis=0)
        best_idx = np.argmax(all_grids, axis=0)

        sinr_grid = np.zeros_like(best_rssi)
        capacity_grid = np.zeros_like(best_rssi)

        rows, cols = best_rssi.shape
        for j in range(rows):
            for i in range(cols):
                serving_idx = best_idx[j, i]
                serving_rssi = all_grids[serving_idx, j, i]

                interferences = []
                for k, sm in enumerate(signal_maps):
                    if k == serving_idx:
                        continue
                    cf = check_channel_interference(routers[serving_idx], routers[k])
                    if cf > 0:
                        interferences.append(all_grids[k, j, i])

                sinr = compute_sinr_db(serving_rssi, interferences, self.noise_floor_dbm)
                sinr_grid[j, i] = sinr

                bw = signal_maps[serving_idx].router.channel_bandwidth_mhz
                capacity_grid[j, i] = shannon_capacity_mbps(bw, sinr)

        return best_rssi, sinr_grid, capacity_grid

    def compare_bands(
        self,
        routers_by_band: dict[str, list[Router]],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """대역별 신호 맵 비교"""
        results = {}
        for band, routers in routers_by_band.items():
            best_rssi, _, _ = self.compute_combined_map(routers, xs, ys)
            results[band] = best_rssi
        return results
