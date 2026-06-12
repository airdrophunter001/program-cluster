"""
3D 신호 히트맵 시각화
- 층별 2D 슬라이스 서브플롯
- 3D Surface 스택 뷰
- 층간 커버리지 비교 바 차트
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from wifi_heatmap.core.building_3d import Building3D
from wifi_heatmap.core.router import Router
from wifi_heatmap.models.path_loss_model_3d import FloorSignalMap

BAND_COLORS = {"2.4GHz": "#4da6ff", "5GHz": "#ff944d", "6GHz": "#bf80ff"}
RSSI_COLORSCALE = [
    [0.0, "#cc0000"], [0.3, "#ff6600"],
    [0.5, "#ffcc00"], [0.7, "#66cc00"], [1.0, "#00cc00"],
]


# ── 층별 히트맵 서브플롯 (Plotly) ────────────────────────────────────

def plot_floor_slices_plotly(
    floor_maps: list[FloorSignalMap],
    building: Building3D,
    routers: list[Router],
    map_type: str = "rssi",   # rssi | sinr | capacity
    threshold_dbm: float = -75.0,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """층별 2D 히트맵을 서브플롯으로 비교"""
    n = len(floor_maps)
    cols = min(n, 3)
    rows = max(1, (n + cols - 1) // cols)

    subplot_titles = [fm.floor_label for fm in floor_maps]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    cfg = {
        "rssi":     {"grid": "rssi_grid",     "colorscale": RSSI_COLORSCALE, "zmin": -100, "zmax": -30,  "unit": "dBm"},
        "sinr":     {"grid": "sinr_grid",     "colorscale": "RdYlGn",        "zmin": 0,    "zmax": 40,   "unit": "dB"},
        "capacity": {"grid": "capacity_grid", "colorscale": "Viridis",       "zmin": 0,    "zmax": 600,  "unit": "Mbps"},
    }[map_type]

    for idx, fm in enumerate(floor_maps):
        row, col = idx // cols + 1, idx % cols + 1
        grid = getattr(fm, cfg["grid"])

        fig.add_trace(go.Heatmap(
            z=grid, x=fm.xs, y=fm.ys,
            colorscale=cfg["colorscale"],
            zmin=cfg["zmin"], zmax=cfg["zmax"],
            colorbar=dict(
                title=cfg["unit"], thickness=12,
                x=1.02 + (col - 1) * 0.01,
            ) if idx == 0 else None,
            showscale=(idx == 0),
            hovertemplate=(
                f"X: %{{x:.1f}}m<br>Y: %{{y:.1f}}m<br>"
                f"{fm.floor_label}<br>{cfg['unit']}: %{{z:.1f}}<extra></extra>"
            ),
        ), row=row, col=col)

        # 벽 오버레이
        if fm.floor_number < len(building.floors):
            floor = building.floors[fm.floor_number]
            for wall in floor.environment.walls:
                lc = "rgba(80,80,80,0.9)" if "concrete" in wall.material.name else "rgba(140,140,140,0.65)"
                lw = 3 if "concrete" in wall.material.name else 1.5
                fig.add_shape(
                    type="line",
                    x0=wall.start[0], y0=wall.start[1],
                    x1=wall.end[0],   y1=wall.end[1],
                    line=dict(color=lc, width=lw),
                    row=row, col=col,
                )

        # AP 마커
        for r in routers:
            if not r.enabled:
                continue
            color = BAND_COLORS.get(r.band, "#4da6ff")
            fig.add_trace(go.Scatter(
                x=[r.x], y=[r.y],
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=11,
                    color=color, line=dict(color="white", width=1.5),
                ),
                name=r.ssid, showlegend=(idx == 0),
                hoverinfo="skip",
            ), row=row, col=col)

        # 커버리지 % 표시
        cov = fm.coverage_at(threshold_dbm)
        cx = (fm.xs[0] + fm.xs[-1]) / 2
        cy = fm.ys[-1] - (fm.ys[-1] - fm.ys[0]) * 0.05
        fig.add_annotation(
            text=f"커버리지 {cov:.0f}%",
            x=cx, y=cy,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.55)",
            row=row, col=col,
        )

    type_title = {"rssi": "RSSI (dBm)", "sinr": "SINR (dB)", "capacity": "이론 용량 (Mbps)"}
    fig.update_layout(
        title=dict(
            text=f"층별 Wi-Fi 신호 비교 — {type_title.get(map_type, map_type)}",
            font=dict(size=16, color="white"),
        ),
        template="plotly_dark",
        height=420 * rows,
        margin=dict(l=40, r=60, t=80, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


# ── 3D 스택 Surface 뷰 ───────────────────────────────────────────────

def plot_3d_stacked(
    floor_maps: list[FloorSignalMap],
    building: Building3D,
    routers: list[Router],
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """층별 신호 슬라이스를 3D로 쌓아서 표시"""
    fig = go.Figure()

    for fm in floor_maps:
        fl = building.floors[fm.floor_number]
        X, Y = np.meshgrid(fm.xs, fm.ys)
        Z = np.full_like(X, fl.z_mid, dtype=float)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=fm.rssi_grid,
            colorscale=RSSI_COLORSCALE,
            cmin=-100, cmax=-30,
            showscale=(fm.floor_number == 0),
            colorbar=dict(
                title="RSSI (dBm)",
                tickvals=[-100, -85, -75, -65, -50, -30],
                ticktext=["-100<br>No Sig", "-85<br>Poor", "-75<br>Fair",
                          "-65<br>Good", "-50<br>Excel", "-30"],
            ) if fm.floor_number == 0 else None,
            opacity=0.80,
            name=fm.floor_label,
            hovertemplate=(
                "X: %{x:.1f}m<br>Y: %{y:.1f}m<br>"
                f"{fm.floor_label}<br>RSSI: %{{surfacecolor:.1f}} dBm<extra></extra>"
            ),
        ))

        # 층 레이블
        fig.add_trace(go.Scatter3d(
            x=[fm.xs.max() + 0.8], y=[fm.ys.max() / 2], z=[fl.z_mid],
            mode="text",
            text=[fm.floor_label],
            textfont=dict(size=13, color="rgba(255,255,255,0.85)"),
            showlegend=False, hoverinfo="skip",
        ))

        # 외벽 윤곽선
        for wall in fl.environment.walls:
            if "concrete" in wall.material.name:
                fig.add_trace(go.Scatter3d(
                    x=[wall.start[0], wall.end[0]],
                    y=[wall.start[1], wall.end[1]],
                    z=[fl.z_mid, fl.z_mid],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.4)", width=3),
                    showlegend=False, hoverinfo="skip",
                ))

    # AP 마커
    for r in routers:
        if not r.enabled:
            continue
        color = BAND_COLORS.get(r.band, "#4da6ff")
        fig.add_trace(go.Scatter3d(
            x=[r.x], y=[r.y], z=[r.height_m],
            mode="markers+text",
            marker=dict(
                symbol="diamond", size=9,
                color=color, line=dict(color="white", width=2),
            ),
            text=[r.ssid], textposition="top right",
            name=f"{r.ssid} ({r.band})",
        ))

    fig.update_layout(
        title=dict(text="3D 신호 강도 (층별 슬라이스)", font=dict(size=16, color="white")),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="rgb(15,15,25)",
                       gridcolor="rgb(50,50,70)", showbackground=True),
            yaxis=dict(title="Y (m)", backgroundcolor="rgb(15,15,25)",
                       gridcolor="rgb(50,50,70)", showbackground=True),
            zaxis=dict(title="높이 (m)", backgroundcolor="rgb(15,15,25)",
                       gridcolor="rgb(50,50,70)", showbackground=True),
            bgcolor="rgb(10,10,20)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.3)),
        ),
        template="plotly_dark",
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    if save_path:
        fig.write_html(str(save_path))
    return fig


# ── 층간 커버리지 비교 (Matplotlib) ──────────────────────────────────

def plot_floor_coverage_bar(
    floor_maps: list[FloorSignalMap],
    threshold_dbm: float = -75.0,
    save_path: Optional[str | Path] = None,
    dpi: int = 130,
) -> plt.Figure:
    """층별 커버리지 및 평균 RSSI 비교 막대 차트"""
    labels = [fm.floor_label for fm in floor_maps]
    coverages = [fm.coverage_at(threshold_dbm) for fm in floor_maps]
    mean_rssi = [
        float(np.mean(fm.rssi_grid[fm.rssi_grid > -99])) for fm in floor_maps
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = [
        "#00cc00" if c >= 90 else "#66cc00" if c >= 75 else "#ffcc00" if c >= 50 else "#ff6600"
        for c in coverages
    ]
    bars = ax1.bar(labels, coverages, color=colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, coverages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.axhline(90, color="white", linestyle="--", linewidth=1, alpha=0.5, label="Target 90%")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title(f"Coverage per Floor (>= {threshold_dbm} dBm)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=9)

    rssi_colors = [
        "#00cc00" if v >= -65 else "#66cc00" if v >= -75 else "#ffcc00" if v >= -85 else "#ff6600"
        for v in mean_rssi
    ]
    bars2 = ax2.bar(labels, mean_rssi, color=rssi_colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars2, mean_rssi):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean RSSI (dBm)")
    ax2.set_title("Mean Signal Strength per Floor")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
