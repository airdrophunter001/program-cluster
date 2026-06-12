"""
커버리지 품질 지표 계산
"""

from typing import Optional

import numpy as np


SIGNAL_QUALITY_LABELS = {
    "excellent": (-50, float("inf")),
    "good":      (-65, -50),
    "fair":      (-75, -65),
    "poor":      (-85, -75),
    "no_signal": (-float("inf"), -85),
}


def compute_coverage_metrics(
    rssi_grid: np.ndarray,
    threshold_dbm: float = -75.0,
    sinr_grid: Optional[np.ndarray] = None,
    min_sinr_db: float = 10.0,
) -> dict:
    """
    전체 커버리지 지표 계산
    Returns: dict with coverage %, RSSI 통계, SINR 통계, 품질 분포
    """
    total = rssi_grid.size
    valid = rssi_grid > -100  # 실제 신호가 있는 셀

    coverage_mask = rssi_grid >= threshold_dbm
    coverage_pct = 100.0 * coverage_mask.sum() / total

    rssi_valid = rssi_grid[valid]
    mean_rssi = float(rssi_valid.mean()) if rssi_valid.size > 0 else -100.0
    min_rssi  = float(rssi_valid.min())  if rssi_valid.size > 0 else -100.0
    max_rssi  = float(rssi_valid.max())  if rssi_valid.size > 0 else -100.0
    std_rssi  = float(rssi_valid.std())  if rssi_valid.size > 0 else 0.0
    median_rssi = float(np.median(rssi_valid)) if rssi_valid.size > 0 else -100.0

    quality_distribution = {}
    for label, (low, high) in SIGNAL_QUALITY_LABELS.items():
        mask = (rssi_grid > low) & (rssi_grid <= high)
        quality_distribution[label] = {
            "count": int(mask.sum()),
            "pct": float(100.0 * mask.sum() / total),
        }

    result = {
        "coverage_pct": coverage_pct,
        "mean_rssi_dbm": mean_rssi,
        "min_rssi_dbm": min_rssi,
        "max_rssi_dbm": max_rssi,
        "std_rssi_db": std_rssi,
        "median_rssi_dbm": median_rssi,
        "threshold_dbm": threshold_dbm,
        "quality_distribution": quality_distribution,
        "total_cells": total,
        "covered_cells": int(coverage_mask.sum()),
    }

    if sinr_grid is not None:
        sinr_valid = sinr_grid[sinr_grid > -50]
        result.update({
            "mean_sinr_db": float(sinr_valid.mean()) if sinr_valid.size > 0 else 0.0,
            "min_sinr_db": float(sinr_valid.min()) if sinr_valid.size > 0 else 0.0,
            "sinr_coverage_pct": float(
                100.0 * (sinr_grid >= min_sinr_db).sum() / total
            ),
        })

    return result


def compute_per_room_metrics(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    rooms: list,
    threshold_dbm: float = -75.0,
) -> list[dict]:
    """방별 커버리지 지표"""
    results = []
    for room in rooms:
        x0, y0, x1, y1 = room.bounds
        xi = np.where((xs >= x0) & (xs <= x1))[0]
        yi = np.where((ys >= y0) & (ys <= y1))[0]

        if len(xi) == 0 or len(yi) == 0:
            continue

        sub_grid = rssi_grid[np.ix_(yi, xi)]
        metrics = compute_coverage_metrics(sub_grid, threshold_dbm)
        metrics["room_id"] = room.id
        metrics["room_name"] = room.name
        results.append(metrics)

    return results


def compute_jain_fairness(rssi_values: np.ndarray) -> float:
    """Jain's Fairness Index: 신호 균등 분포 지표 (0~1, 1이 최적)"""
    valid = rssi_values[rssi_values > -100]
    if len(valid) == 0:
        return 0.0
    # dBm → 정규화된 선형 값
    normalized = (valid - valid.min()) / (valid.max() - valid.min() + 1e-9)
    n = len(normalized)
    return float((normalized.sum() ** 2) / (n * (normalized ** 2).sum() + 1e-9))


def compute_overlap_ratio(
    signal_maps: list[np.ndarray],
    threshold_dbm: float = -75.0,
) -> float:
    """다중 AP 간 커버리지 중복 비율"""
    if len(signal_maps) < 2:
        return 0.0
    coverage_masks = [m >= threshold_dbm for m in signal_maps]
    overlap = np.ones_like(coverage_masks[0], dtype=bool)
    for mask in coverage_masks:
        overlap &= mask
    any_coverage = np.zeros_like(coverage_masks[0], dtype=bool)
    for mask in coverage_masks:
        any_coverage |= mask
    if any_coverage.sum() == 0:
        return 0.0
    return float(100.0 * overlap.sum() / any_coverage.sum())


def format_metrics_table(metrics: dict) -> str:
    """메트릭을 보기 좋은 텍스트 표로 변환"""
    lines = [
        "=" * 50,
        "  Wi-Fi Coverage Metrics Report",
        "=" * 50,
        f"  Coverage     : {metrics['coverage_pct']:.1f}% (≥ {metrics['threshold_dbm']} dBm)",
        f"  Mean RSSI    : {metrics['mean_rssi_dbm']:.1f} dBm",
        f"  Min RSSI     : {metrics['min_rssi_dbm']:.1f} dBm",
        f"  Max RSSI     : {metrics['max_rssi_dbm']:.1f} dBm",
        f"  Std Dev      : {metrics['std_rssi_db']:.1f} dB",
        "",
        "  Quality Distribution:",
    ]
    for label, data in metrics.get("quality_distribution", {}).items():
        lines.append(f"    {label:<12}: {data['pct']:5.1f}%  ({data['count']} cells)")
    if "mean_sinr_db" in metrics:
        lines.append("")
        lines.append(f"  Mean SINR    : {metrics['mean_sinr_db']:.1f} dB")
        lines.append(f"  SINR Cov.    : {metrics['sinr_coverage_pct']:.1f}%")
    lines.append("=" * 50)
    return "\n".join(lines)
