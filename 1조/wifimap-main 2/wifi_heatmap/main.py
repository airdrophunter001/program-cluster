"""
Wi-Fi 신호 강도 맵 시뮬레이터 — CLI 진입점

사용법:
  python -m wifi_heatmap.main                        # 기본 시뮬레이션
  python -m wifi_heatmap.main --optimize             # GA 최적화 포함
  python -m wifi_heatmap.main --band 5GHz            # 대역 지정
  python -m wifi_heatmap.main --model fspl            # 전파 모델 지정
  python -m wifi_heatmap.main --compare-bands        # 대역 비교
  python -m wifi_heatmap.main --export-pdf           # PDF 리포트 생성
  streamlit run wifi_heatmap/visualization/interactive.py  # 대시보드
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 패키지 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wifi_heatmap.core.propagation import PropagationModel, PropagationParams
from wifi_heatmap.core.environment import BuildingEnvironment
from wifi_heatmap.core.router import Router, create_default_routers
from wifi_heatmap.models.path_loss_model import PathLossModel
from wifi_heatmap.visualization.heatmap import (
    plot_heatmap_matplotlib,
    plot_heatmap_plotly,
    plot_band_comparison,
    plot_sinr_map,
)
from wifi_heatmap.visualization.overlay import (
    render_coverage_zones,
    generate_coverage_summary_chart,
)
from wifi_heatmap.optimizer.placement import GeneticOptimizer, GAConfig, plot_optimization_history
from wifi_heatmap.utils.metrics import compute_coverage_metrics, format_metrics_table
from wifi_heatmap.utils.export import export_csv, export_pdf
from wifi_heatmap.utils.logger import get_logger

logger = get_logger("wifi_heatmap.main")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    res = cfg["simulation"]["resolution"]
    w = cfg["simulation"]["grid_width"]
    h = cfg["simulation"]["grid_height"]
    return np.arange(0, w + res, res), np.arange(0, h + res, res)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wi-Fi Signal Strength Map Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=Path,
                   default=Path(__file__).parent / "config.yaml",
                   help="설정 파일 경로")
    p.add_argument("--floor-plan", type=Path,
                   default=Path(__file__).parent / "data/floor_plans/sample_office.json",
                   help="도면 JSON 파일 경로")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="결과 저장 디렉토리")
    p.add_argument("--band", choices=["2.4GHz", "5GHz", "6GHz"], default="2.4GHz",
                   help="주파수 대역")
    p.add_argument("--model", choices=["fspl", "log_distance", "itu_r"], default="log_distance",
                   help="전파 모델")
    p.add_argument("--pl-exponent", type=float, default=None,
                   help="경로 손실 지수 (log_distance 모델)")
    p.add_argument("--no-walls", action="store_true",
                   help="벽 감쇠 비적용")
    p.add_argument("--optimize", action="store_true",
                   help="유전 알고리즘으로 AP 배치 최적화")
    p.add_argument("--num-aps", type=int, default=3,
                   help="시뮬레이션 AP 수")
    p.add_argument("--compare-bands", action="store_true",
                   help="대역 비교 차트 생성")
    p.add_argument("--export-pdf", action="store_true",
                   help="PDF 리포트 생성")
    p.add_argument("--export-csv", action="store_true", default=True,
                   help="CSV 리포트 생성 (기본 활성화)")
    p.add_argument("--plotly", action="store_true",
                   help="Plotly HTML 히트맵 생성")
    p.add_argument("--threshold", type=float, default=-75.0,
                   help="커버리지 임계값 (dBm)")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Wi-Fi Signal Strength Simulator ===")
    logger.info(f"Band: {args.band} | Model: {args.model} | Walls: {not args.no_walls}")

    # 설정 로드
    cfg = load_config(args.config)
    xs, ys = build_grid(cfg)

    # 환경 로드
    logger.info(f"Loading floor plan: {args.floor_plan}")
    env = BuildingEnvironment.from_json(
        args.floor_plan,
        extra_materials=cfg.get("materials"),
    )
    logger.info(f"  Loaded: {env.name} ({env.width}m x {env.height}m), {len(env.walls)} walls")

    # 전파 모델 파라미터
    pl_exp = args.pl_exponent or cfg["propagation"]["path_loss_exponent"]
    params = PropagationParams(
        model=PropagationModel(args.model),
        path_loss_exponent=pl_exp,
        reference_distance=cfg["propagation"]["reference_distance"],
        shadowing_std_db=cfg["propagation"]["shadowing_std_db"],
    )

    # AP 설정
    if args.optimize:
        logger.info(f"Running genetic algorithm optimization ({args.num_aps} APs)...")
        ga_cfg = GAConfig(
            population_size=cfg["optimizer"]["population_size"],
            generations=cfg["optimizer"]["generations"],
            mutation_rate=cfg["optimizer"]["mutation_rate"],
            crossover_rate=cfg["optimizer"]["crossover_rate"],
            elite_size=cfg["optimizer"]["elite_size"],
            target_coverage_pct=cfg["optimizer"]["target_coverage_pct"],
            min_signal_dbm=cfg["optimizer"]["min_signal_dbm"],
        )
        optimizer = GeneticOptimizer(
            environment=env,
            propagation_params=params,
            ga_config=ga_cfg,
            band=args.band,
            noise_floor_dbm=cfg["sinr"]["noise_floor_dbm"],
        )
        result = optimizer.optimize(
            num_aps=args.num_aps,
            xs=xs, ys=ys,
            verbose=True,
            use_walls=not args.no_walls,
        )
        routers = result.best_routers
        logger.info(f"Optimization complete: {result.summary()}")

        fig_history = plot_optimization_history(result)
        history_path = args.output_dir / "optimization_history.png"
        fig_history.savefig(str(history_path), dpi=150, bbox_inches="tight")
        plt.close(fig_history)
        logger.info(f"Saved: {history_path}")
    else:
        routers = create_default_routers(band=args.band)
        routers = routers[:args.num_aps]

    logger.info(f"Using {len(routers)} AP(s):")
    for r in routers:
        logger.info(f"  [{r.ssid}] pos=({r.x:.1f},{r.y:.1f}) band={r.band} tx={r.tx_power_dbm}dBm ch={r.channel}")

    # 신호 맵 계산
    logger.info("Computing signal maps...")
    model = PathLossModel(env, params, noise_floor_dbm=cfg["sinr"]["noise_floor_dbm"])
    best_rssi, sinr_grid, capacity_grid = model.compute_combined_map(
        routers, xs, ys, use_wall_attenuation=not args.no_walls
    )

    # 지표 출력
    metrics = compute_coverage_metrics(best_rssi, args.threshold, sinr_grid)
    print("\n" + format_metrics_table(metrics))

    # Matplotlib 히트맵
    logger.info("Generating heatmap (matplotlib)...")
    heatmap_path = args.output_dir / f"heatmap_{args.band.replace('/', '_')}.png"
    fig = plot_heatmap_matplotlib(
        best_rssi, xs, ys, env, routers,
        title=f"Wi-Fi Signal Heatmap — {args.band} ({args.model})",
        save_path=heatmap_path,
        dpi=cfg["visualization"]["dpi"],
    )
    plt.close(fig)
    logger.info(f"Saved: {heatmap_path}")

    # SINR 맵
    sinr_path = args.output_dir / "sinr_map.png"
    fig_sinr = plot_sinr_map(sinr_grid, xs, ys, env, routers, save_path=sinr_path)
    plt.close(fig_sinr)
    logger.info(f"Saved: {sinr_path}")

    # 커버리지 존 맵
    zones_path = args.output_dir / "coverage_zones.png"
    fig_zones = render_coverage_zones(best_rssi, xs, ys, env, routers, save_path=zones_path)
    plt.close(fig_zones)
    logger.info(f"Saved: {zones_path}")

    # 요약 차트
    summary_path = args.output_dir / "coverage_summary.png"
    fig_summary = generate_coverage_summary_chart(best_rssi, save_path=summary_path)
    plt.close(fig_summary)
    logger.info(f"Saved: {summary_path}")

    # 대역 비교
    band_comparison_path = None
    if args.compare_bands:
        logger.info("Generating band comparison chart...")
        band_grids = {}
        for b in ["2.4GHz", "5GHz", "6GHz"]:
            test_routers = [
                Router(x=r.x, y=r.y, band=b, tx_power_dbm=r.tx_power_dbm, ssid=r.ssid)
                for r in routers
            ]
            test_model = PathLossModel(env, params, cfg["sinr"]["noise_floor_dbm"])
            band_best, _, _ = test_model.compute_combined_map(test_routers, xs, ys, not args.no_walls)
            band_grids[b] = band_best
        band_comparison_path = args.output_dir / "band_comparison.png"
        fig_bc = plot_band_comparison(band_grids, xs, ys, env, save_path=band_comparison_path)
        plt.close(fig_bc)
        logger.info(f"Saved: {band_comparison_path}")

    # Plotly HTML
    if args.plotly:
        plotly_path = args.output_dir / "heatmap_interactive.html"
        fig_plotly = plot_heatmap_plotly(
            best_rssi, xs, ys, env, routers,
            title=f"Wi-Fi Signal Heatmap — {args.band}",
            save_path=plotly_path,
        )
        logger.info(f"Saved: {plotly_path}")

    # CSV 내보내기
    if args.export_csv:
        csv_path = args.output_dir / cfg["export"]["csv_filename"]
        export_csv(
            best_rssi, xs, ys, routers, env, csv_path,
            threshold_dbm=args.threshold,
            sinr_grid=sinr_grid,
        )
        logger.info(f"Saved: {csv_path}")

    # PDF 리포트
    if args.export_pdf:
        try:
            pdf_path = args.output_dir / cfg["export"]["pdf_filename"]
            export_pdf(
                best_rssi, xs, ys, routers, env, pdf_path,
                threshold_dbm=args.threshold,
                sinr_grid=sinr_grid,
                heatmap_image_path=heatmap_path,
                band_comparison_image_path=band_comparison_path,
            )
            logger.info(f"Saved: {pdf_path}")
        except ImportError as e:
            logger.warning(f"PDF export skipped: {e}")

    logger.info(f"\nAll outputs saved to: {args.output_dir.resolve()}")
    logger.info("To launch interactive dashboard:")
    logger.info("  streamlit run wifi_heatmap/visualization/interactive.py")


if __name__ == "__main__":
    main()
