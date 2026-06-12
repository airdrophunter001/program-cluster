"""
도면 이미지 위에 히트맵 오버레이 렌더링
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from wifi_heatmap.core.environment import BuildingEnvironment
from wifi_heatmap.core.router import Router


def overlay_on_floorplan(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    floorplan_image_path: str | Path,
    routers: list[Router],
    alpha: float = 0.6,
    vmin: float = -100,
    vmax: float = -30,
    title: str = "Wi-Fi Heatmap Overlay",
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """도면 이미지 위에 히트맵 오버레이"""
    img = Image.open(floorplan_image_path).convert("RGBA")
    img_arr = np.array(img)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_arr, extent=[xs[0], xs[-1], ys[-1], ys[0]], aspect="equal")

    colors = ["#cc0000", "#ff6600", "#ffcc00", "#66cc00", "#00cc00"]
    cmap = mcolors.LinearSegmentedColormap.from_list("wifi", colors, N=256)

    im = ax.imshow(
        rssi_grid,
        extent=[xs[0], xs[-1], ys[-1], ys[0]],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        aspect="equal",
        interpolation="bilinear",
    )

    for router in routers:
        if router.enabled:
            ax.plot(router.x, router.y, "^", color="#ffffff",
                    markersize=14, markeredgecolor="#333333",
                    markeredgewidth=1.5, zorder=10)
            ax.annotate(
                router.ssid, (router.x, router.y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333", alpha=0.7),
                zorder=11,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("RSSI (dBm)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def render_coverage_zones(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    environment: BuildingEnvironment,
    routers: list[Router],
    thresholds: Optional[list[tuple[float, str, str]]] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """신호 강도 구간별 커버리지 영역 시각화"""
    if thresholds is None:
        thresholds = [
            (-50, "Excellent (> -50 dBm)", "#00cc00"),
            (-65, "Good (-65 ~ -50 dBm)",  "#66cc00"),
            (-75, "Fair (-75 ~ -65 dBm)",  "#ffcc00"),
            (-85, "Poor (-85 ~ -75 dBm)",  "#ff6600"),
            (-100, "No Signal (< -85 dBm)", "#cc0000"),
        ]

    fig, ax = plt.subplots(figsize=(11, 8))
    extent = [xs[0], xs[-1], ys[-1], ys[0]]

    # 배경
    ax.set_facecolor("#222222")

    # 구간별 컬러 마스크
    for i, (thresh, label, color) in enumerate(thresholds):
        if i == 0:
            mask = rssi_grid >= thresh
        else:
            prev_thresh = thresholds[i - 1][0]
            mask = (rssi_grid < prev_thresh) & (rssi_grid >= thresh)

        colored = np.zeros((*rssi_grid.shape, 4), dtype=np.float32)
        r, g, b = mcolors.to_rgb(color)
        colored[mask] = [r, g, b, 0.8]
        colored[~mask, 3] = 0.0
        ax.imshow(colored, extent=extent, origin="upper", aspect="equal")

    # 벽
    for wall in environment.walls:
        lc = "#ffffff" if "concrete" in wall.material.name else "#aaaaaa"
        lw = 3 if "concrete" in wall.material.name else 1.5
        ax.plot(
            [wall.start[0], wall.end[0]],
            [wall.start[1], wall.end[1]],
            color=lc, linewidth=lw,
        )

    # AP
    for router in routers:
        if router.enabled:
            ax.plot(router.x, router.y, "^", color="white",
                    markersize=13, markeredgecolor="#333", zorder=10)

    # 방 레이블
    for room in environment.rooms:
        cx = (room.bounds[0] + room.bounds[2]) / 2
        cy = (room.bounds[1] + room.bounds[3]) / 2
        ax.text(cx, cy, room.name, ha="center", va="center",
                fontsize=8, color="white", alpha=0.6,
                bbox=dict(facecolor="none", edgecolor="none"))

    # 범례
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
        for _, label, color in thresholds
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=9, framealpha=0.85)

    ax.set_title("Wi-Fi Coverage Zones", fontsize=13, fontweight="bold", color="white")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[-1], ys[0])
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", facecolor="#222222")
    return fig


def generate_coverage_summary_chart(
    rssi_grid: np.ndarray,
    title: str = "Coverage Distribution",
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """커버리지 분포 요약 파이/막대 차트"""
    thresholds = [
        (-50,  "Excellent\n(≥-50)",  "#00cc00"),
        (-65,  "Good\n(-65~-50)",    "#66cc00"),
        (-75,  "Fair\n(-75~-65)",    "#ffcc00"),
        (-85,  "Poor\n(-85~-75)",    "#ff6600"),
        (-1000, "No Sig\n(<-85)",    "#cc0000"),
    ]

    total = rssi_grid.size
    sizes, labels, colors = [], [], []

    prev = 0
    for thresh, label, color in thresholds:
        if prev == 0:
            count = (rssi_grid >= thresh).sum()
        else:
            count = ((rssi_grid < prev) & (rssi_grid >= thresh)).sum()
        pct = 100 * count / total
        if pct > 0.1:
            sizes.append(pct)
            labels.append(f"{label}\n{pct:.1f}%")
            colors.append(color)
        prev = thresh

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Coverage Pie Chart")

    level_names = [t[1].split("\n")[0] for t in thresholds]
    bar_sizes = []
    prev = 0
    for thresh, _, _ in thresholds:
        if prev == 0:
            count = (rssi_grid >= thresh).sum()
        else:
            count = ((rssi_grid < prev) & (rssi_grid >= thresh)).sum()
        bar_sizes.append(100 * count / total)
        prev = thresh

    bars = ax2.bar(level_names, bar_sizes,
                   color=[t[2] for t in thresholds], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, bar_sizes):
        if val > 0.5:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Coverage Bar Chart")
    ax2.set_ylim(0, max(bar_sizes) * 1.15 if bar_sizes else 100)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig
