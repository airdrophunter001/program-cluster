"""
Plotly 기반 3D 건물 모델 시각화
- 층별 벽면 메시 + 바닥 슬래브
- AP 위치 마커
- 카메라 프리셋
"""

from __future__ import annotations

import plotly.graph_objects as go

from wifi_heatmap.core.building_3d import Building3D
from wifi_heatmap.core.router import Router

BAND_COLORS = {"2.4GHz": "#4da6ff", "5GHz": "#ff944d", "6GHz": "#bf80ff"}


def build_model_figure(
    building: Building3D,
    routers: list[Router] | None = None,
    title: str = "3D 건물 모델",
    show_outer_only: bool = False,
) -> go.Figure:
    """
    3D 건물 모델 Plotly Figure 생성

    Parameters
    ----------
    building : Building3D
    routers  : AP 리스트 (없으면 생략)
    title    : 차트 제목
    show_outer_only : True면 외벽(concrete)만 표시
    """
    fig = go.Figure()

    # ── 벽 메시 ─────────────────────────────────────────────────────
    for mesh in building.get_mesh_data():
        name: str = mesh["name"]
        is_floor = name.endswith("_floor")

        # 외벽만 표시 옵션
        if show_outer_only and not is_floor:
            # floor label 뒤의 재질 이름 확인
            mat = mesh["text"].split("|")[-1].strip() if "|" in mesh["text"] else ""
            if "concrete" not in mat:
                continue

        fig.add_trace(go.Mesh3d(
            x=mesh["x"], y=mesh["y"], z=mesh["z"],
            i=mesh["i"], j=mesh["j"], k=mesh["k"],
            color=mesh["color"],
            flatshading=True,
            showscale=False,
            hoverinfo="text",
            text=mesh["text"],
            name=name,
            showlegend=False,
            lighting=dict(ambient=0.65, diffuse=0.85, specular=0.1, roughness=0.8),
            lightposition=dict(x=1, y=1, z=5),
        ))

    # ── 층 번호 레이블 ───────────────────────────────────────────────
    for fl in building.floors:
        fig.add_trace(go.Scatter3d(
            x=[building.footprint_width / 2],
            y=[building.footprint_height + 0.3],
            z=[fl.z_mid],
            mode="text",
            text=[fl.label],
            textfont=dict(size=13, color="rgba(255,255,255,0.75)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 층 경계선 (투명 사각형 윤곽)
    w = building.footprint_width
    h = building.footprint_height
    for fl in building.floors:
        for z in (fl.z_bottom, fl.z_top):
            fig.add_trace(go.Scatter3d(
                x=[0, w, w, 0, 0],
                y=[0, 0, h, h, 0],
                z=[z] * 5,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── AP 마커 ──────────────────────────────────────────────────────
    if routers:
        for r in routers:
            if not r.enabled:
                continue
            color = BAND_COLORS.get(r.band, "#4da6ff")
            fig.add_trace(go.Scatter3d(
                x=[r.x], y=[r.y], z=[r.height_m],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color=color,
                    line=dict(color="white", width=2),
                ),
                text=[r.ssid],
                textposition="top right",
                textfont=dict(size=10, color="white"),
                name=f"{r.ssid} ({r.band})",
                hovertemplate=(
                    f"<b>{r.ssid}</b><br>"
                    f"Band: {r.band}<br>"
                    f"Pos: ({r.x:.1f}, {r.y:.1f}, {r.height_m:.1f} m)<br>"
                    f"TX: {r.tx_power_dbm} dBm<extra></extra>"
                ),
            ))

    # ── 레이아웃 ─────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="white")),
        scene=dict(
            xaxis=dict(
                title="X (m)",
                backgroundcolor="rgb(18,18,28)",
                gridcolor="rgb(55,55,75)",
                zerolinecolor="rgb(55,55,75)",
                showbackground=True,
            ),
            yaxis=dict(
                title="Y (m)",
                backgroundcolor="rgb(18,18,28)",
                gridcolor="rgb(55,55,75)",
                zerolinecolor="rgb(55,55,75)",
                showbackground=True,
            ),
            zaxis=dict(
                title="높이 (m)",
                backgroundcolor="rgb(18,18,28)",
                gridcolor="rgb(55,55,75)",
                zerolinecolor="rgb(55,55,75)",
                showbackground=True,
            ),
            bgcolor="rgb(12,12,22)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.1),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            x=0.01, y=0.99,
        ),
    )
    return fig


def build_model_with_signal_preview(
    building: Building3D,
    routers: list[Router],
    signal_grids: dict[int, "np.ndarray"],  # floor_number → rssi_grid
    xs: "np.ndarray",
    ys: "np.ndarray",
    vmin: float = -100,
    vmax: float = -30,
) -> go.Figure:
    """3D 모델 + 층별 신호 슬라이스 동시 표시"""
    import numpy as np

    fig = build_model_figure(building, routers, title="3D 건물 + 신호 히트맵")

    colorscale = [
        [0.0, "#cc0000"], [0.3, "#ff6600"],
        [0.5, "#ffcc00"], [0.7, "#66cc00"], [1.0, "#00cc00"],
    ]

    for floor_num, grid in signal_grids.items():
        if floor_num >= len(building.floors):
            continue
        fl = building.floors[floor_num]
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, fl.z_mid + 0.05, dtype=float)  # 슬라이스를 바닥에서 살짝 위로

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=grid,
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            showscale=(floor_num == 0),
            colorbar=dict(
                title="RSSI (dBm)",
                tickvals=[-100, -85, -75, -65, -50, -30],
            ) if floor_num == 0 else None,
            opacity=0.75,
            name=f"Signal {fl.label}",
            hovertemplate=(
                "X: %{x:.1f}m<br>Y: %{y:.1f}m<br>"
                f"{fl.label}<br>RSSI: %{{surfacecolor:.1f}} dBm<extra></extra>"
            ),
        ))

    return fig
