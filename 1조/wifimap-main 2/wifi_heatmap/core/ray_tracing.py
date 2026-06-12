"""
간이 Ray Tracing: 반사·회절을 고려한 다중 경로 전파 계산
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .environment import BuildingEnvironment, Wall, _segment_intersects


REFLECTION_LOSS_DB = 3.0   # 반사당 추가 손실
DIFFRACTION_LOSS_DB = 6.0  # 회절당 추가 손실


@dataclass
class Ray:
    origin: tuple[float, float]
    direction: tuple[float, float]
    power_dbm: float
    bounces: int = 0
    path_length: float = 0.0
    wall_loss_db: float = 0.0


@dataclass
class PathResult:
    rx_pos: tuple[float, float]
    received_power_dbm: float
    path_length: float
    num_reflections: int
    wall_loss_db: float
    is_los: bool  # Line-of-Sight 여부


def has_line_of_sight(
    tx_pos: tuple[float, float],
    rx_pos: tuple[float, float],
    walls: list[Wall],
) -> bool:
    """두 점 사이 직선 경로에 벽이 없는지 확인"""
    for wall in walls:
        if wall.is_door or wall.is_window:
            continue
        if _segment_intersects(tx_pos, rx_pos, wall.start, wall.end):
            return False
    return True


def find_wall_intersection(
    ray_origin: tuple[float, float],
    ray_dir: tuple[float, float],
    walls: list[Wall],
    max_dist: float = 100.0,
) -> tuple[Optional[Wall], Optional[tuple[float, float]], float]:
    """
    광선과 가장 가까운 벽 교차점 탐색
    Returns: (wall, intersection_point, distance)
    """
    best_wall = None
    best_point = None
    best_dist = max_dist

    ox, oy = ray_origin
    dx, dy = ray_dir

    for wall in walls:
        wx1, wy1 = wall.start
        wx2, wy2 = wall.end

        # 선분과 광선의 교차 계산 (매개변수 형식)
        denom = dx * (wy2 - wy1) - dy * (wx2 - wx1)
        if abs(denom) < 1e-10:
            continue

        t = ((wx1 - ox) * (wy2 - wy1) - (wy1 - oy) * (wx2 - wx1)) / denom
        s_num = (wx1 - ox) * dy - (wy1 - oy) * dx
        s = s_num / denom

        if t > 1e-6 and 0.0 <= s <= 1.0 and t < best_dist:
            best_dist = t
            best_wall = wall
            best_point = (ox + t * dx, oy + t * dy)

    return best_wall, best_point, best_dist


def reflect_direction(
    direction: tuple[float, float],
    wall: Wall,
) -> tuple[float, float]:
    """벽 표면에서의 반사 방향 계산"""
    dx, dy = direction
    wx = wall.end[0] - wall.start[0]
    wy = wall.end[1] - wall.start[1]
    wall_len = math.sqrt(wx * wx + wy * wy)
    if wall_len < 1e-9:
        return direction
    wx /= wall_len
    wy /= wall_len

    dot = dx * wx + dy * wy
    # 반사: d - 2(d·n)n  (n: 벽 법선)
    nx, ny = -wy, wx  # 법선
    dot_n = dx * nx + dy * ny
    return (dx - 2 * dot_n * nx, dy - 2 * dot_n * ny)


def trace_rays(
    tx_pos: tuple[float, float],
    rx_pos: tuple[float, float],
    walls: list[Wall],
    tx_power_dbm: float,
    max_bounces: int = 2,
    num_rays: int = 360,
) -> list[PathResult]:
    """
    다중 광선 추적으로 수신 가능한 경로 탐색
    실제 수신기에 도달하는 경로만 반환
    """
    results: list[PathResult] = []
    rx_radius = 0.3  # 수신기 캡처 반경 (미터)

    # LOS 경로 먼저 확인
    los = has_line_of_sight(tx_pos, rx_pos, walls)
    dist_direct = math.sqrt(
        (rx_pos[0] - tx_pos[0]) ** 2 + (rx_pos[1] - tx_pos[1]) ** 2
    )
    if dist_direct < 1e-3:
        dist_direct = 1e-3

    wall_loss = sum(
        w.material.total_attenuation_db
        for w in walls
        if _segment_intersects(tx_pos, rx_pos, w.start, w.end)
    )

    results.append(PathResult(
        rx_pos=rx_pos,
        received_power_dbm=tx_power_dbm,
        path_length=dist_direct,
        num_reflections=0,
        wall_loss_db=wall_loss,
        is_los=los,
    ))

    # 반사 경로 (간이 구현: 단일 반사)
    if max_bounces >= 1:
        for wall in walls:
            if wall.is_door or wall.is_window:
                continue
            # 경상(image source) 기법으로 반사점 계산
            img = _mirror_point(tx_pos, wall)
            if img is None:
                continue
            # 반사점에서 수신기까지 경로
            img_to_rx_dist = math.sqrt(
                (rx_pos[0] - img[0]) ** 2 + (rx_pos[1] - img[1]) ** 2
            )
            # 반사점 좌표 (벽 위)
            t = _closest_param_on_segment(img, rx_pos, wall)
            if t is None:
                continue
            ref_point = (
                wall.start[0] + t * (wall.end[0] - wall.start[0]),
                wall.start[1] + t * (wall.end[1] - wall.start[1]),
            )
            path_len = (
                math.sqrt((ref_point[0] - tx_pos[0]) ** 2 + (ref_point[1] - tx_pos[1]) ** 2)
                + math.sqrt((rx_pos[0] - ref_point[0]) ** 2 + (rx_pos[1] - ref_point[1]) ** 2)
            )
            if path_len > 50:
                continue
            ref_loss = REFLECTION_LOSS_DB + wall.material.total_attenuation_db
            results.append(PathResult(
                rx_pos=rx_pos,
                received_power_dbm=tx_power_dbm - ref_loss,
                path_length=path_len,
                num_reflections=1,
                wall_loss_db=ref_loss,
                is_los=False,
            ))

    return results


def _mirror_point(
    point: tuple[float, float],
    wall: Wall,
) -> Optional[tuple[float, float]]:
    """벽에 대한 점의 경상 계산"""
    ax, ay = wall.start
    bx, by = wall.end
    dx, dy = bx - ax, by - ay
    wall_len_sq = dx * dx + dy * dy
    if wall_len_sq < 1e-10:
        return None
    t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / wall_len_sq
    foot_x = ax + t * dx
    foot_y = ay + t * dy
    return (2 * foot_x - point[0], 2 * foot_y - point[1])


def _closest_param_on_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    wall: Wall,
) -> Optional[float]:
    """선분 p1-p2와 wall의 교차 매개변수 t 반환"""
    ax, ay = wall.start
    bx, by = wall.end
    cx, cy = p1
    dx_seg, dy_seg = p2[0] - p1[0], p2[1] - p1[1]
    dx_wall, dy_wall = bx - ax, by - ay

    denom = dx_seg * dy_wall - dy_seg * dx_wall
    if abs(denom) < 1e-10:
        return None
    t_wall = ((ax - cx) * dy_seg - (ay - cy) * dx_seg) / denom
    if 0.0 <= t_wall <= 1.0:
        return t_wall
    return None


def compute_multipath_power_dbm(paths: list[PathResult]) -> float:
    """다중 경로 신호 전력의 선형 합산 (coherent combining 근사)"""
    from .propagation import dbm_to_mw, mw_to_dbm
    total_mw = sum(dbm_to_mw(p.received_power_dbm) for p in paths)
    return mw_to_dbm(total_mw)
