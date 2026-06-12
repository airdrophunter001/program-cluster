"""
다층 건물 3D 모델

Floor: z_bottom ~ z_top 구간의 단일 층 (BuildingEnvironment 포함)
Building3D: 여러 Floor를 쌓아 올린 3D 건물

3D 신호 경로 감쇠:
  - 수평 분량 → 각 층의 2D 벽 감쇠
  - 수직 분량 → 층간 바닥/천장 관통 감쇠 (floor_ceiling_attenuation_db per floor)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wifi_heatmap.core.environment import BuildingEnvironment


# 재료별 층간 감쇠 기본값 (ITU-R P.1238 참고)
FLOOR_ATTENUATION_DB = {
    "concrete": 18.0,   # 콘크리트 슬래브
    "wood":     10.0,   # 목조 바닥
    "light":     7.0,   # 경량 슬래브
}


@dataclass
class Floor:
    number: int                       # 0 = 1층
    z_bottom: float                   # m
    z_top: float                      # m
    environment: BuildingEnvironment

    @property
    def height(self) -> float:
        return self.z_top - self.z_bottom

    @property
    def z_mid(self) -> float:
        return (self.z_bottom + self.z_top) / 2

    @property
    def label(self) -> str:
        return f"F{self.number + 1}"


@dataclass
class Building3D:
    """다층 건물 3D 모델"""
    name: str
    floors: list[Floor] = field(default_factory=list)
    floor_material: str = "concrete"   # 층간 슬래브 재질

    # ── 층 관리 ──────────────────────────────────────────────────────
    @property
    def num_floors(self) -> int:
        return len(self.floors)

    @property
    def total_height_m(self) -> float:
        return self.floors[-1].z_top if self.floors else 0.0

    @property
    def footprint_width(self) -> float:
        return self.floors[0].environment.width if self.floors else 0.0

    @property
    def footprint_height(self) -> float:
        return self.floors[0].environment.height if self.floors else 0.0

    def add_floor(self, env: BuildingEnvironment, floor_h: float = 3.0) -> "Building3D":
        z0 = self.floors[-1].z_top if self.floors else 0.0
        self.floors.append(Floor(
            number=len(self.floors),
            z_bottom=z0,
            z_top=z0 + floor_h,
            environment=env,
        ))
        return self

    def get_floor_at_z(self, z: float) -> Optional[Floor]:
        for fl in self.floors:
            if fl.z_bottom <= z <= fl.z_top:
                return fl
        # 범위 밖이면 가장 가까운 층 반환
        if self.floors:
            if z < self.floors[0].z_bottom:
                return self.floors[0]
            return self.floors[-1]
        return None

    # ── 3D 경로 감쇠 ─────────────────────────────────────────────────
    def wall_attenuation_3d(
        self,
        tx: tuple[float, float, float],
        rx: tuple[float, float, float],
    ) -> float:
        """
        3D 경로 (tx→rx) 의 총 벽·층간 감쇠 계산

        알고리즘:
        1. 수직 이동 → 관통 층 수 × 층간감쇠
        2. 각 관련 층에서 2D 수평 벽 감쇠 (층 기여 비율 가중)
        """
        tx_fl = self.get_floor_at_z(tx[2])
        rx_fl = self.get_floor_at_z(rx[2])

        # ① 층간 감쇠
        attn = 0.0
        if tx_fl and rx_fl:
            floors_crossed = abs(rx_fl.number - tx_fl.number)
            per_floor = FLOOR_ATTENUATION_DB.get(self.floor_material, 15.0)
            attn += floors_crossed * per_floor

        # ② 관련 층들의 수평 벽 감쇠
        tx_2d = (tx[0], tx[1])
        rx_2d = (rx[0], rx[1])

        if tx_fl and rx_fl:
            fn_min = min(tx_fl.number, rx_fl.number)
            fn_max = max(tx_fl.number, rx_fl.number)
            involved = list(range(fn_min, fn_max + 1))
        elif tx_fl:
            involved = [tx_fl.number]
        elif rx_fl:
            involved = [rx_fl.number]
        else:
            involved = []

        if involved:
            weight = 1.0 / len(involved)
            for fn in involved:
                if fn < len(self.floors):
                    fl_att = self.floors[fn].environment.compute_wall_attenuation(tx_2d, rx_2d)
                    attn += weight * fl_att

        return attn

    # ── Plotly 메시 데이터 ────────────────────────────────────────────
    def get_mesh_data(self) -> list[dict]:
        """
        벽마다 dict(x,y,z,i,j,k,color,name) 반환
        Plotly go.Mesh3d에 직접 사용
        """
        MAT_COLOR = {
            "concrete":  "rgba(120,115,110,0.85)",
            "brick":     "rgba(160,82,45,0.80)",
            "drywall":   "rgba(215,215,215,0.65)",
            "glass":     "rgba(135,206,235,0.35)",
            "door_wood": "rgba(139,69,19,0.75)",
            "metal":     "rgba(180,180,180,0.90)",
            "wood":      "rgba(205,133,63,0.75)",
        }
        DEFAULT = "rgba(200,200,200,0.60)"

        meshes = []
        for fl in self.floors:
            z0, z1 = fl.z_bottom, fl.z_top

            # 바닥 슬래브
            w = fl.environment.width
            h = fl.environment.height
            meshes.append({
                "x": [0, w, w, 0],
                "y": [0, 0, h, h],
                "z": [z0, z0, z0, z0],
                "i": [0, 0],
                "j": [1, 2],
                "k": [2, 3],
                "color": "rgba(240,235,210,0.25)",
                "name": f"{fl.label}_floor",
                "text": f"{fl.label} 바닥",
            })

            for wall in fl.environment.walls:
                x1, y1 = wall.start
                x2, y2 = wall.end
                color = MAT_COLOR.get(wall.material.name, DEFAULT)
                meshes.append({
                    "x": [x1, x2, x2, x1],
                    "y": [y1, y2, y2, y1],
                    "z": [z0, z0, z1, z1],
                    "i": [0, 0],
                    "j": [1, 2],
                    "k": [2, 3],
                    "color": color,
                    "name": f"{fl.label}_{wall.id}",
                    "text": f"{fl.label} | {wall.material.name}",
                })

        return meshes


# ── 팩토리 함수 ──────────────────────────────────────────────────────

def build_from_single_env(
    env: BuildingEnvironment,
    num_floors: int = 1,
    floor_height: float = 3.0,
    floor_material: str = "concrete",
) -> Building3D:
    """단일 평면도를 복제해 다층 건물 생성"""
    b = Building3D(name=env.name, floor_material=floor_material)
    for _ in range(num_floors):
        b.add_floor(deepcopy(env), floor_height)
    return b


def build_from_env_list(
    envs: list[BuildingEnvironment],
    floor_height: float = 3.0,
    floor_material: str = "concrete",
) -> Building3D:
    """층마다 다른 평면도를 가진 건물 생성"""
    name = envs[0].name if envs else "Building"
    b = Building3D(name=name, floor_material=floor_material)
    for env in envs:
        b.add_floor(env, floor_height)
    return b
