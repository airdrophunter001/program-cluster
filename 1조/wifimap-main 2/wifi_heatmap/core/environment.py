"""
건물 환경: 벽·문·창문 파싱 및 재질별 전파 감쇠 계산
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Material:
    name: str
    attenuation_db_per_m: float
    thickness_m: float

    @property
    def total_attenuation_db(self) -> float:
        return self.attenuation_db_per_m * self.thickness_m


DEFAULT_MATERIALS: dict[str, Material] = {
    "concrete": Material("concrete", 12.0, 0.20),
    "brick": Material("brick", 8.0, 0.15),
    "drywall": Material("drywall", 3.0, 0.10),
    "glass": Material("glass", 2.0, 0.006),
    "wood": Material("wood", 4.0, 0.05),
    "metal": Material("metal", 25.0, 0.003),
    "door_wood": Material("door_wood", 3.0, 0.04),
}


@dataclass
class Wall:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    material: Material
    is_door: bool = False
    is_window: bool = False

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def as_segment(self) -> tuple:
        return (self.start, self.end)


@dataclass
class Room:
    id: str
    name: str
    bounds: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

    def contains(self, x: float, y: float) -> bool:
        return (
            self.bounds[0] <= x <= self.bounds[2]
            and self.bounds[1] <= y <= self.bounds[3]
        )


@dataclass
class BuildingEnvironment:
    name: str
    width: float
    height: float
    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    materials: dict[str, Material] = field(default_factory=lambda: dict(DEFAULT_MATERIALS))

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        extra_materials: Optional[dict] = None,
    ) -> "BuildingEnvironment":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        materials = dict(DEFAULT_MATERIALS)
        if extra_materials:
            for name, cfg in extra_materials.items():
                materials[name] = Material(
                    name=name,
                    attenuation_db_per_m=cfg["attenuation_db_per_m"],
                    thickness_m=cfg["thickness_m"],
                )

        env = cls(
            name=data.get("name", "Building"),
            width=data["width"],
            height=data["height"],
            materials=materials,
        )

        # 문/창문 위치 수집 (벽 파싱 시 열린 구간 처리)
        door_positions: dict[str, list] = {}
        window_positions: dict[str, list] = {}

        for d in data.get("doors", []):
            wid = d["wall_id"]
            door_positions.setdefault(wid, []).append(d)
        for w in data.get("windows", []):
            wid = w["wall_id"]
            window_positions.setdefault(wid, []).append(w)

        for w_data in data.get("walls", []):
            mat_name = w_data.get("material", "drywall")
            mat = materials.get(mat_name, DEFAULT_MATERIALS["drywall"])
            wall = Wall(
                id=w_data["id"],
                start=tuple(w_data["start"]),
                end=tuple(w_data["end"]),
                material=mat,
            )
            env.walls.append(wall)

            # 문 세그먼트 추가
            for d in door_positions.get(w_data["id"], []):
                door_mat = materials.get(d.get("material", "door_wood"), DEFAULT_MATERIALS["door_wood"])
                env.walls.append(Wall(
                    id=d["id"],
                    start=(d["position"][0] - d["width"] / 2, d["position"][1]),
                    end=(d["position"][0] + d["width"] / 2, d["position"][1]),
                    material=door_mat,
                    is_door=True,
                ))

            # 창문 세그먼트 추가
            for win in window_positions.get(w_data["id"], []):
                win_mat = materials.get(win.get("material", "glass"), DEFAULT_MATERIALS["glass"])
                env.walls.append(Wall(
                    id=win["id"],
                    start=(win["position"][0] - win["width"] / 2, win["position"][1]),
                    end=(win["position"][0] + win["width"] / 2, win["position"][1]),
                    material=win_mat,
                    is_window=True,
                ))

        for r_data in data.get("rooms", []):
            env.rooms.append(Room(
                id=r_data["id"],
                name=r_data["name"],
                bounds=tuple(r_data["bounds"]),
            ))

        return env

    def compute_wall_attenuation(
        self,
        tx_pos: tuple[float, float],
        rx_pos: tuple[float, float],
    ) -> float:
        """송수신 경로상의 모든 벽 감쇠 합산"""
        total_attenuation = 0.0
        for wall in self.walls:
            if _segment_intersects(tx_pos, rx_pos, wall.start, wall.end):
                total_attenuation += wall.material.total_attenuation_db
        return total_attenuation

    def get_room_at(self, x: float, y: float) -> Optional[Room]:
        for room in self.rooms:
            if room.contains(x, y):
                return room
        return None


def _cross_product_2d(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, q, r) -> bool:
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def _segment_intersects(
    p1: tuple, p2: tuple,
    p3: tuple, p4: tuple,
    eps: float = 1e-9,
) -> bool:
    """두 선분의 교차 여부 판별"""
    d1 = _cross_product_2d(p3, p4, p1)
    d2 = _cross_product_2d(p3, p4, p2)
    d3 = _cross_product_2d(p1, p2, p3)
    d4 = _cross_product_2d(p1, p2, p4)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    if abs(d1) < eps and _on_segment(p3, p1, p4):
        return True
    if abs(d2) < eps and _on_segment(p3, p2, p4):
        return True
    if abs(d3) < eps and _on_segment(p1, p3, p2):
        return True
    if abs(d4) < eps and _on_segment(p1, p4, p2):
        return True

    return False


def precompute_wall_attenuation_grid(
    env: BuildingEnvironment,
    router_pos: tuple[float, float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """그리드 전체에 대한 벽 감쇠 사전 계산"""
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = env.compute_wall_attenuation(router_pos, (x, y))
    return grid
