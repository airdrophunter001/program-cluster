"""
Matplotlib + Plotly 히트맵 시각화
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px

from wifi_heatmap.core.environment import BuildingEnvironment, Wall
from wifi_heatmap.core.router import Router


SIGNAL_LEVELS = [
    (-50, "Excellent", "#00cc00"),
    (-65, "Good",      "#66cc00"),
    (-75, "Fair",      "#ffcc00"),
    (-85, "Poor",      "#ff6600"),
    (-100, "No Signal", "#cc0000"),
]


def _get_signal_cmap() -> mcolors.LinearSegmentedColormap:
    colors = ["#cc0000", "#ff6600", "#ffcc00", "#66cc00", "#00cc00"]
    return mcolors.LinearSegmentedColormap.from_list("wifi_signal", colors, N=256)


def plot_heatmap_matplotlib(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    environment: BuildingEnvironment,
    routers: list[Router],
    title: str = "Wi-Fi Signal Strength Heatmap",
    vmin: float = -100,
    vmax: float = -30,
    show_contours: bool = True,
    show_walls: bool = True,
    contour_levels: int = 8,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = _get_signal_cmap()
    extent = [xs[0], xs[-1], ys[-1], ys[0]]
    im = ax.imshow(
        rssi_grid,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
        alpha=0.85,
    )

    # 등고선
    if show_contours:
        X, Y = np.meshgrid(xs, ys)
        levels = np.linspace(vmin, vmax, contour_levels)
        cs = ax.contour(X, Y, rssi_grid, levels=levels, colors="white", alpha=0.4, linewidths=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f dBm")

    # 신호 강도 경계선 (주요 임계값)
    threshold_levels = [-50, -65, -75, -85]
    threshold_colors = ["#00cc00", "#66cc00", "#ffcc00", "#ff6600"]
    X, Y = np.meshgrid(xs, ys)
    for lvl, col in zip(threshold_levels, threshold_colors):
        cs2 = ax.contour(X, Y, rssi_grid, levels=[lvl], colors=[col], linewidths=1.5, alpha=0.9)
        ax.clabel(cs2, fmt=f"{lvl} dBm", fontsize=8)

    # 벽 그리기
    if show_walls and environment:
        for wall in environment.walls:
            if wall.is_window:
                lc, lw, ls = "#87CEEB", 1.5, "--"
            elif wall.is_door:
                lc, lw, ls = "#8B4513", 1.5, ":"
            else:
                mat = wall.material.name
                if "concrete" in mat:
                    lc, lw, ls = "#555555", 3.0, "-"
                elif "brick" in mat:
                    lc, lw, ls = "#A0522D", 2.5, "-"
                else:
                    lc, lw, ls = "#999999", 2.0, "-"
            ax.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                color=lc, linewidth=lw, linestyle=ls, solid_capstyle="round",
            )

    # 공유기 위치 표시
    for router in routers:
        if not router.enabled:
            continue
        band_colors = {"2.4GHz": "#1f77b4", "5GHz": "#ff7f0e", "6GHz": "#9467bd"}
        color = band_colors.get(router.band, "#1f77b4")
        ax.plot(router.x, router.y, "^", color=color, markersize=14,
                markeredgecolor="white", markeredgewidth=1.5, zorder=10)
        ax.annotate(
            f"{router.ssid}\n({router.band})",
            (router.x, router.y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            zorder=11,
        )

    # 컬러바
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Signal Strength (dBm)", fontsize=10)
    cbar.set_ticks([-100, -85, -75, -65, -50, -30])
    for tick, (_, label, color) in zip(cbar.ax.get_yticklabels(), SIGNAL_LEVELS[::-1] + [(-30, "", "")]):
        pass

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[-1], ys[0])
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # 범례
    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=3, label=f"{lvl} dBm ({name})")
        for lvl, name, c in SIGNAL_LEVELS[:-1]
    ]
    band_colors = {"2.4GHz": "#1f77b4", "5GHz": "#ff7f0e", "6GHz": "#9467bd"}
    legend_elements += [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=c,
                   markersize=10, label=f"{band} AP")
        for band, c in band_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8, ncol=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_heatmap_plotly(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    environment: BuildingEnvironment,
    routers: list[Router],
    title: str = "Wi-Fi Signal Strength Heatmap",
    vmin: float = -100,
    vmax: float = -30,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    colorscale = [
        [0.0,  "#cc0000"],
        [0.3,  "#ff6600"],
        [0.5,  "#ffcc00"],
        [0.7,  "#66cc00"],
        [1.0,  "#00cc00"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=rssi_grid,
        x=xs,
        y=ys,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title="dBm",
            tickvals=[-100, -85, -75, -65, -50, -30],
            ticktext=["-100 (No Signal)", "-85 (Poor)", "-75 (Fair)",
                      "-65 (Good)", "-50 (Excellent)", "-30"],
        ),
        hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>RSSI: %{z:.1f} dBm<extra></extra>",
    ))

    # 등고선
    fig.add_trace(go.Contour(
        z=rssi_grid,
        x=xs,
        y=ys,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color="white"),
            start=vmin, end=vmax, size=10,
        ),
        line=dict(width=1, color="rgba(255,255,255,0.5)"),
        opacity=0.3,
    ))

    # 벽 그리기
    if environment:
        for wall in environment.walls:
            color = "rgba(100,100,100,0.8)"
            width = 2
            if "concrete" in wall.material.name:
                color = "rgba(60,60,60,0.9)"
                width = 4
            elif wall.is_window:
                color = "rgba(135,206,235,0.7)"
                width = 2
            elif wall.is_door:
                color = "rgba(139,69,19,0.7)"
                width = 2
            fig.add_shape(
                type="line",
                x0=wall.start[0], y0=wall.start[1],
                x1=wall.end[0], y1=wall.end[1],
                line=dict(color=color, width=width),
            )

    # 공유기 위치
    band_colors = {"2.4GHz": "#1f77b4", "5GHz": "#ff7f0e", "6GHz": "#9467bd"}
    for router in routers:
        if not router.enabled:
            continue
        fig.add_trace(go.Scatter(
            x=[router.x], y=[router.y],
            mode="markers+text",
            marker=dict(
                symbol="triangle-up",
                size=16,
                color=band_colors.get(router.band, "#1f77b4"),
                line=dict(color="white", width=2),
            ),
            text=[router.ssid],
            textposition="top right",
            textfont=dict(size=10, color="white"),
            name=f"{router.ssid} ({router.band})",
            hovertemplate=(
                f"AP: {router.ssid}<br>Band: {router.band}<br>"
                f"TX: {router.tx_power_dbm} dBm<br>"
                f"Pos: ({router.x:.1f}, {router.y:.1f})<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="X (m)", scaleanchor="y"),
        yaxis=dict(title="Y (m)", autorange="reversed"),
        height=600,
        template="plotly_dark",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def plot_band_comparison(
    band_grids: dict[str, np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    environment: BuildingEnvironment,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """대역별 신호 맵 비교 (서브플롯)"""
    n = len(band_grids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    cmap = _get_signal_cmap()
    extent = [xs[0], xs[-1], ys[-1], ys[0]]

    for ax, (band, grid) in zip(axes, band_grids.items()):
        im = ax.imshow(
            grid, extent=extent, origin="upper",
            cmap=cmap, vmin=-100, vmax=-30,
            aspect="equal", interpolation="bilinear",
        )
        if environment:
            for wall in environment.walls:
                ax.plot(
                    [wall.start[0], wall.end[0]],
                    [wall.start[1], wall.end[1]],
                    color="#555", linewidth=1.5,
                )
        ax.set_title(f"{band}", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="dBm")

    axes[0].set_ylabel("Y (m)")
    fig.suptitle("Wi-Fi Band Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_sinr_map(
    sinr_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    environment: BuildingEnvironment,
    routers: list[Router],
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    extent = [xs[0], xs[-1], ys[-1], ys[0]]
    cmap = plt.cm.RdYlGn
    im = ax.imshow(
        sinr_grid, extent=extent, origin="upper",
        cmap=cmap, vmin=0, vmax=40,
        aspect="equal", interpolation="bilinear",
    )
    X, Y = np.meshgrid(xs, ys)
    cs = ax.contour(X, Y, sinr_grid, levels=[10, 20, 30], colors="white",
                    alpha=0.6, linewidths=1.0)
    ax.clabel(cs, fmt="%.0f dB", fontsize=8)

    if environment:
        for wall in environment.walls:
            ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                    color="#333", linewidth=2.0)
    for router in routers:
        if router.enabled:
            ax.plot(router.x, router.y, "^", color="#1f77b4",
                    markersize=12, markeredgecolor="white", zorder=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("SINR (dB)", fontsize=10)
    ax.set_title("SINR Map", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[-1], ys[0])
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
