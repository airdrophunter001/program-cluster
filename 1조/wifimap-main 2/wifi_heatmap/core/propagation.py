"""
Wi-Fi 전파 손실 모델: FSPL, Log-distance, ITU-R P.1238
"""

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


SPEED_OF_LIGHT = 3e8


class PropagationModel(Enum):
    FSPL = "fspl"
    LOG_DISTANCE = "log_distance"
    ITU_R = "itu_r"


@dataclass
class PropagationParams:
    model: PropagationModel = PropagationModel.LOG_DISTANCE
    path_loss_exponent: float = 3.0
    reference_distance: float = 1.0
    shadowing_std_db: float = 4.0
    floor_count: int = 0
    apply_shadowing: bool = False


def fspl_db(distance_m: float, frequency_hz: float) -> float:
    """자유공간 경로 손실 (Free-Space Path Loss)"""
    if distance_m <= 0:
        return 0.0
    wavelength = SPEED_OF_LIGHT / frequency_hz
    loss = 20 * math.log10(4 * math.pi * distance_m / wavelength)
    return loss


def fspl_at_reference(frequency_hz: float, reference_distance: float = 1.0) -> float:
    """기준 거리에서의 FSPL"""
    return fspl_db(reference_distance, frequency_hz)


def log_distance_path_loss(
    distance_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 3.0,
    reference_distance: float = 1.0,
) -> float:
    """Log-distance 경로 손실 모델 (실내 환경)"""
    if distance_m <= 0:
        return 0.0
    if distance_m < reference_distance:
        distance_m = reference_distance

    pl_ref = fspl_db(reference_distance, frequency_hz)
    pl = pl_ref + 10 * path_loss_exponent * math.log10(distance_m / reference_distance)
    return pl


def itu_r_path_loss(
    distance_m: float,
    frequency_hz: float,
    num_floors: int = 0,
    floor_penetration_factor: Optional[float] = None,
) -> float:
    """
    ITU-R P.1238 실내 전파 손실 모델
    L = 20*log10(f) + N*log10(d) + Lf(n) - 28  [dB]
    f: MHz, d: 미터, N: 거리 손실 계수, Lf: 층간 손실
    """
    if distance_m <= 0:
        return 0.0

    freq_mhz = frequency_hz / 1e6

    if freq_mhz < 2500:
        N = 28
    elif freq_mhz < 5500:
        N = 31
    else:
        N = 32

    # 층간 손실 계수 (ITU-R P.1238 Table 3)
    if floor_penetration_factor is not None:
        Lf = floor_penetration_factor
    else:
        if freq_mhz < 2500:
            Lf = 4 * num_floors if num_floors > 0 else 0
        elif freq_mhz < 5500:
            Lf = 15 + 4 * (num_floors - 1) if num_floors > 0 else 0
        else:
            Lf = 15 + 4 * (num_floors - 1) if num_floors > 0 else 0

    path_loss = 20 * math.log10(freq_mhz) + N * math.log10(distance_m) + Lf - 28
    return max(path_loss, 0.0)


def compute_path_loss(
    distance_m: float,
    frequency_hz: float,
    params: PropagationParams,
) -> float:
    """선택된 모델로 경로 손실 계산"""
    if params.model == PropagationModel.FSPL:
        pl = fspl_db(distance_m, frequency_hz)
    elif params.model == PropagationModel.LOG_DISTANCE:
        pl = log_distance_path_loss(
            distance_m,
            frequency_hz,
            params.path_loss_exponent,
            params.reference_distance,
        )
    elif params.model == PropagationModel.ITU_R:
        pl = itu_r_path_loss(distance_m, frequency_hz, params.floor_count)
    else:
        raise ValueError(f"Unknown model: {params.model}")

    if params.apply_shadowing and params.shadowing_std_db > 0:
        pl += np.random.normal(0, params.shadowing_std_db)

    return pl


def received_power_dbm(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    path_loss_db: float,
    wall_loss_db: float = 0.0,
) -> float:
    """수신 전력 계산 (Friis 공식 확장)"""
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db - wall_loss_db


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10)


def mw_to_dbm(mw: float) -> float:
    if mw <= 0:
        return -float("inf")
    return 10 * math.log10(mw)


def combine_signals_dbm(signals_dbm: list[float]) -> float:
    """여러 신호를 선형 합산 후 dBm으로 변환"""
    if not signals_dbm:
        return -float("inf")
    total_mw = sum(dbm_to_mw(s) for s in signals_dbm)
    return mw_to_dbm(total_mw)


def compute_sinr_db(
    signal_dbm: float,
    interference_dbm_list: list[float],
    noise_floor_dbm: float = -100.0,
) -> float:
    """SINR (Signal-to-Interference-plus-Noise Ratio) 계산"""
    signal_mw = dbm_to_mw(signal_dbm)
    noise_mw = dbm_to_mw(noise_floor_dbm)
    interference_mw = sum(dbm_to_mw(i) for i in interference_dbm_list)
    sinr = signal_mw / (interference_mw + noise_mw)
    return 10 * math.log10(sinr) if sinr > 0 else -float("inf")


def shannon_capacity_mbps(
    bandwidth_mhz: float,
    sinr_db: float,
) -> float:
    """Shannon-Hartley 이론적 최대 처리량"""
    sinr_linear = 10 ** (sinr_db / 10)
    capacity_bps = bandwidth_mhz * 1e6 * math.log2(1 + sinr_linear)
    return capacity_bps / 1e6
