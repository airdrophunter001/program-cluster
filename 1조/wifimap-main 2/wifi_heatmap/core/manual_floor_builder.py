"""
수동 평면도 빌더
- 방(사각형) 목록을 입력받아 BuildingEnvironment JSON(dict) 형식으로 변환
- 이미지 인식이 어려운 표/그리드 형태의 복잡한 평면도(여러 동/여러 실)를
  좌표 기반으로 직접 정의할 때 사용
"""

from __future__ import annotations


def rooms_to_floor_json(
    name: str,
    width: float,
    height: float,
    rooms: list[dict],
    outer_material: str = "concrete",
    inner_material: str = "drywall",
) -> dict:
    """방 사각형 목록 → BuildingEnvironment JSON dict 변환

    rooms: [{"name": str, "x": float, "y": float, "w": float, "h": float}, ...]
    각 방의 4변을 벽 세그먼트로 변환하고, 두 방이 공유하는 변은 하나로 합칩니다.
    건물 외곽(x=0, x=width, y=0, y=height)에 위치한 변은 외벽(outer_material),
    그 외의 변은 내벽(inner_material)으로 분류합니다.
    """
    eps = 1e-3
    segments: dict[tuple, dict] = {}

    def add_segment(p1: tuple[float, float], p2: tuple[float, float]) -> None:
        a = (round(p1[0], 3), round(p1[1], 3))
        b = (round(p2[0], 3), round(p2[1], 3))
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key not in segments:
            segments[key] = {"start": key[0], "end": key[1]}

    for r in rooms:
        x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for i in range(4):
            add_segment(corners[i], corners[(i + 1) % 4])

    walls = []
    for i, seg in enumerate(segments.values()):
        (x1, y1), (x2, y2) = seg["start"], seg["end"]
        on_boundary = (
            (abs(x1) < eps and abs(x2) < eps)
            or (abs(x1 - width) < eps and abs(x2 - width) < eps)
            or (abs(y1) < eps and abs(y2) < eps)
            or (abs(y1 - height) < eps and abs(y2 - height) < eps)
        )
        material = outer_material if on_boundary else inner_material
        walls.append({
            "id": f"w{i + 1}",
            "start": [x1, y1],
            "end": [x2, y2],
            "material": material,
            "is_door": False,
            "is_window": False,
        })

    room_defs = [
        {
            "id": f"room{i + 1}",
            "name": r["name"],
            "bounds": [
                float(r["x"]), float(r["y"]),
                float(r["x"]) + float(r["w"]), float(r["y"]) + float(r["h"]),
            ],
        }
        for i, r in enumerate(rooms)
    ]

    return {
        "name": name,
        "width": width,
        "height": height,
        "walls": walls,
        "rooms": room_defs,
    }
