"""
커버리지 리포트 CSV / PDF 출력
"""

import csv
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np

from wifi_heatmap.utils.metrics import compute_coverage_metrics, compute_per_room_metrics


def export_csv(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    routers: list,
    environment,
    output_path: str | Path,
    threshold_dbm: float = -75.0,
    sinr_grid: Optional[np.ndarray] = None,
    capacity_grid: Optional[np.ndarray] = None,
) -> Path:
    """커버리지 데이터를 CSV로 출력"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Wi-Fi Coverage Report", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        # 공유기 정보
        writer.writerow(["=== Access Points ==="])
        writer.writerow(["ID", "SSID", "X (m)", "Y (m)", "Band", "TX Power (dBm)", "Channel", "Enabled"])
        for r in routers:
            writer.writerow([r.router_id, r.ssid, r.x, r.y, r.band,
                             r.tx_power_dbm, r.channel, r.enabled])
        writer.writerow([])

        # 전체 지표
        metrics = compute_coverage_metrics(rssi_grid, threshold_dbm, sinr_grid)
        writer.writerow(["=== Overall Metrics ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Coverage (%)", f"{metrics['coverage_pct']:.2f}"])
        writer.writerow(["Mean RSSI (dBm)", f"{metrics['mean_rssi_dbm']:.2f}"])
        writer.writerow(["Min RSSI (dBm)", f"{metrics['min_rssi_dbm']:.2f}"])
        writer.writerow(["Max RSSI (dBm)", f"{metrics['max_rssi_dbm']:.2f}"])
        writer.writerow(["Std Dev (dB)", f"{metrics['std_rssi_db']:.2f}"])
        writer.writerow(["Threshold (dBm)", f"{threshold_dbm}"])
        writer.writerow(["Total Cells", metrics['total_cells']])
        writer.writerow(["Covered Cells", metrics['covered_cells']])
        if "mean_sinr_db" in metrics:
            writer.writerow(["Mean SINR (dB)", f"{metrics['mean_sinr_db']:.2f}"])
        writer.writerow([])

        # 품질 분포
        writer.writerow(["=== Quality Distribution ==="])
        writer.writerow(["Level", "Count", "Percentage (%)"])
        for label, data in metrics["quality_distribution"].items():
            writer.writerow([label, data["count"], f"{data['pct']:.2f}"])
        writer.writerow([])

        # 방별 지표
        if environment and environment.rooms:
            room_metrics = compute_per_room_metrics(
                rssi_grid, xs, ys, environment.rooms, threshold_dbm
            )
            writer.writerow(["=== Per-Room Metrics ==="])
            writer.writerow(["Room", "Coverage (%)", "Mean RSSI (dBm)", "Min RSSI (dBm)"])
            for rm in room_metrics:
                writer.writerow([
                    rm["room_name"],
                    f"{rm['coverage_pct']:.2f}",
                    f"{rm['mean_rssi_dbm']:.2f}",
                    f"{rm['min_rssi_dbm']:.2f}",
                ])
            writer.writerow([])

        # 그리드 데이터
        writer.writerow(["=== Grid Data (RSSI dBm) ==="])
        header = ["Y\\X"] + [f"{x:.1f}" for x in xs]
        writer.writerow(header)
        for j, y in enumerate(ys):
            row = [f"{y:.1f}"] + [f"{rssi_grid[j, i]:.1f}" for i in range(len(xs))]
            writer.writerow(row)

    return output_path


def export_pdf(
    rssi_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    routers: list,
    environment,
    output_path: str | Path,
    threshold_dbm: float = -75.0,
    sinr_grid: Optional[np.ndarray] = None,
    heatmap_image_path: Optional[str | Path] = None,
    band_comparison_image_path: Optional[str | Path] = None,
) -> Path:
    """커버리지 리포트를 PDF로 출력 (reportlab 사용)"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("reportlab 패키지가 필요합니다: pip install reportlab")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, spaceAfter=12)
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=8)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
    normal_style = styles["Normal"]

    metrics = compute_coverage_metrics(rssi_grid, threshold_dbm, sinr_grid)
    story = []

    # 표지
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("📡 Wi-Fi Coverage Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        normal_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
    story.append(Spacer(1, 0.5 * cm))

    # 요약 지표
    story.append(Paragraph("1. Summary Metrics", h1_style))
    summary_data = [
        ["Metric", "Value"],
        ["Overall Coverage", f"{metrics['coverage_pct']:.1f}%"],
        ["Threshold", f"{threshold_dbm} dBm"],
        ["Mean RSSI", f"{metrics['mean_rssi_dbm']:.1f} dBm"],
        ["Min RSSI", f"{metrics['min_rssi_dbm']:.1f} dBm"],
        ["Max RSSI", f"{metrics['max_rssi_dbm']:.1f} dBm"],
        ["Std Dev", f"{metrics['std_rssi_db']:.1f} dB"],
    ]
    if "mean_sinr_db" in metrics:
        summary_data.append(["Mean SINR", f"{metrics['mean_sinr_db']:.1f} dB"])
        summary_data.append(["SINR Coverage", f"{metrics['sinr_coverage_pct']:.1f}%"])

    t = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # 품질 분포
    story.append(Paragraph("2. Signal Quality Distribution", h1_style))
    qual_data = [["Quality Level", "Cell Count", "Coverage (%)"]]
    labels_map = {
        "excellent": "Excellent (≥ -50 dBm)",
        "good":      "Good (-65 ~ -50 dBm)",
        "fair":      "Fair (-75 ~ -65 dBm)",
        "poor":      "Poor (-85 ~ -75 dBm)",
        "no_signal": "No Signal (< -85 dBm)",
    }
    qual_colors = [
        colors.HexColor("#00cc00"),
        colors.HexColor("#66cc00"),
        colors.HexColor("#ffcc00"),
        colors.HexColor("#ff6600"),
        colors.HexColor("#cc0000"),
    ]
    for (k, label), qcolor in zip(labels_map.items(), qual_colors):
        data = metrics["quality_distribution"].get(k, {"count": 0, "pct": 0})
        qual_data.append([label, str(data["count"]), f"{data['pct']:.1f}%"])

    qt = Table(qual_data, colWidths=[10 * cm, 4 * cm, 4 * cm])
    qt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(qt)
    story.append(Spacer(1, 0.5 * cm))

    # AP 정보
    story.append(Paragraph("3. Access Point Configuration", h1_style))
    ap_data = [["SSID", "Band", "Position", "TX Power", "Channel"]]
    for r in routers:
        ap_data.append([
            r.ssid, r.band,
            f"({r.x:.1f}, {r.y:.1f})",
            f"{r.tx_power_dbm} dBm",
            str(r.channel),
        ])
    at = Table(ap_data, colWidths=[4*cm, 3*cm, 4*cm, 3.5*cm, 3.5*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(at)

    # 히트맵 이미지 삽입
    if heatmap_image_path and Path(heatmap_image_path).exists():
        story.append(PageBreak())
        story.append(Paragraph("4. Signal Heatmap", h1_style))
        img = RLImage(str(heatmap_image_path), width=16 * cm, height=10 * cm)
        story.append(img)

    if band_comparison_image_path and Path(band_comparison_image_path).exists():
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("5. Band Comparison", h1_style))
        img2 = RLImage(str(band_comparison_image_path), width=16 * cm, height=6 * cm)
        story.append(img2)

    doc.build(story)
    return output_path
