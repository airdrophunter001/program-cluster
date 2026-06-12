"""
Wi-Fi 신호 시뮬레이터 — Streamlit 인터랙티브 대시보드
실행: streamlit run wifi_heatmap/visualization/interactive.py

4단계 마법사 파이프라인:
  Step 1. 평면도 선택 (고정 평면도 / 수동 입력)
  Step 2. 구조 확인 / 수동 조정 → 3D 모델 파라미터 설정
  Step 3. 3D 건물 모델 미리보기 → 확인
  Step 4. AP 배치 → 시뮬레이션 → 3D 히트맵 결과
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wifi_heatmap.core.propagation import PropagationModel, PropagationParams
from wifi_heatmap.core.environment import BuildingEnvironment
from wifi_heatmap.core.building_3d import Building3D, build_from_single_env
from wifi_heatmap.core.router import Router, BAND_DEFAULTS
from wifi_heatmap.core.building_3d import build_from_env_list
from wifi_heatmap.core.manual_floor_builder import rooms_to_floor_json
from wifi_heatmap.optimizer.big_data_search import run_big_data_search
from wifi_heatmap.models.path_loss_model import PathLossModel
from wifi_heatmap.models.path_loss_model_3d import PathLossModel3D
from wifi_heatmap.visualization.model_3d import build_model_figure, build_model_with_signal_preview
from wifi_heatmap.visualization.heatmap_3d import (
    plot_floor_slices_plotly,
    plot_3d_stacked,
    plot_floor_coverage_bar,
)
from wifi_heatmap.visualization.heatmap import plot_heatmap_plotly
from wifi_heatmap.utils.metrics import compute_coverage_metrics

# ── 경로 상수 ────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "wifi_heatmap" / "config.yaml"
FIXED_FLOOR_PLAN = ROOT / "wifi_heatmap" / "data" / "floor_plans" / "fixed_school_building.json"


# ── 설정 로드 ────────────────────────────────────────────────────────
@st.cache_data
def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def _load_fixed_env() -> dict:
    with open(FIXED_FLOOR_PLAN, encoding="utf-8") as f:
        return json.load(f)


def _build_grid(width: float, height: float, resolution: float):
    xs = np.arange(0, width + resolution, resolution)
    ys = np.arange(0, height + resolution, resolution)
    return xs, ys


# ── 세션 상태 초기화 ─────────────────────────────────────────────────
def _init_state():
    defaults = {
        "step": 1,
        "floor_json": None,         # 단일 층 도면 dict (고정/수동 단층)
        "floor_json_list": None,    # 층별 도면 dict 리스트 (다층 수동 입력)
        "building": None,           # Building3D
        "floor_maps": None,         # list[FloorSignalMap]
        "routers": [],
        "sim_done": False,
        "coverage_summary": None,
        "bigdata_result": None,
        "bigdata_band": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _step_indicator(current: int):
    labels = ["1 평면도 선택", "2 구조 확인", "3 3D 모델", "4 시뮬레이션", "5 빅데이터 분석"]
    cols = st.columns(5)
    for i, (col, label) in enumerate(zip(cols, labels), 1):
        style = "**" if i == current else ""
        icon = "✅" if i < current else ("🔵" if i == current else "⚪")
        col.markdown(f"{icon} {style}{label}{style}", unsafe_allow_html=False)
    st.divider()


# ════════════════════════════════════════════════════════════════════
# STEP 1: 평면도 선택
# ════════════════════════════════════════════════════════════════════
def step1_upload():
    st.header("📐 Step 1: 평면도 선택")

    source = st.radio(
        "도면 소스",
        ["고정 평면도 (5층 교사동)", "수동 입력 (방 배치 표)"],
        horizontal=True,
    )

    if source == "수동 입력 (방 배치 표)":
        _step1_manual()
        return

    # ── 고정 평면도 ──────────────────────────────────────────────────
    data = _load_fixed_env()
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.info(
            f"**{data['name']}** — {data['width']}m × {data['height']}m, "
            f"5층 건물 (층고 4.5m, 외벽 콘크리트 / 내벽 드라이월)을 사용합니다."
        )
        st.plotly_chart(_render_2d_floorplan(data), use_container_width=True)
        if st.button("📂 이 평면도로 시작", type="primary"):
            st.session_state.floor_json = data
            st.session_state.floor_json_list = None
            st.session_state.step = 2
            st.rerun()
    with col_right:
        st.json({
            "name": data["name"],
            "크기": f"{data['width']} × {data['height']} m",
            "벽 수": len(data["walls"]),
            "방 수": len(data["rooms"]),
        })


# ════════════════════════════════════════════════════════════════════
# STEP 1 (대안): 방 배치 표 수동 입력
# ════════════════════════════════════════════════════════════════════
def _step1_manual():
    """이미지 인식이 어려운 표/그리드 형태 평면도를 방 좌표 직접 입력으로 변환"""
    st.subheader("🏗️ 수동 입력 (방 배치 표)")
    st.caption(
        "방 이름과 위치(x, y), 크기(width, height)를 표에 직접 입력하면 "
        "각 방의 외곽선을 벽으로 변환합니다. 인접한 방이 공유하는 변은 "
        "하나의 벽(내벽)으로 합쳐지고, 건물 바깥쪽 경계는 외벽(콘크리트)으로 처리됩니다. "
        "도면의 그림 비율과 실제 치수(m)를 참고해 좌표를 입력하세요."
    )

    num_floors = st.number_input("층 수", 1, 10, 1, key="manual_num_floors")

    floor_json_list: list[dict] = []
    tabs = st.tabs([f"Floor {i + 1}" for i in range(int(num_floors))])

    for i, tab in enumerate(tabs):
        with tab:
            col_in, col_prev = st.columns([1, 1.4])

            with col_in:
                fname = st.text_input("층 이름", f"Floor {i + 1}", key=f"manual_fname_{i}")
                fw = st.number_input("건물 폭 (m)", 1.0, 500.0, 30.0, 0.5, key=f"manual_fw_{i}")
                fh = st.number_input("건물 깊이 (m)", 1.0, 500.0, 20.0, 0.5, key=f"manual_fh_{i}")

                room_key = f"manual_rooms_{i}"
                if room_key not in st.session_state:
                    st.session_state[room_key] = pd.DataFrame([
                        {"name": "방1", "x": 0.0, "y": 0.0, "w": 5.0, "h": 5.0},
                        {"name": "방2", "x": 5.0, "y": 0.0, "w": 5.0, "h": 5.0},
                    ])

                st.markdown("**방 목록** (행 추가/삭제 가능)")
                edited_df = st.data_editor(
                    st.session_state[room_key],
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    key=f"manual_editor_{i}",
                    column_config={
                        "name": st.column_config.TextColumn("이름", required=True),
                        "x": st.column_config.NumberColumn("x (m)", min_value=0.0, step=0.1, required=True),
                        "y": st.column_config.NumberColumn("y (m)", min_value=0.0, step=0.1, required=True),
                        "w": st.column_config.NumberColumn("width (m)", min_value=0.1, step=0.1, required=True),
                        "h": st.column_config.NumberColumn("height (m)", min_value=0.1, step=0.1, required=True),
                    },
                )
                st.session_state[room_key] = edited_df

            rooms = [
                r for r in edited_df.to_dict("records")
                if r.get("name") and r.get("w", 0) and r.get("h", 0)
            ]

            with col_prev:
                if rooms:
                    fj = rooms_to_floor_json(fname, float(fw), float(fh), rooms)
                else:
                    fj = {"name": fname, "width": float(fw), "height": float(fh), "walls": [], "rooms": []}
                st.markdown("**미리보기**")
                st.plotly_chart(_render_2d_floorplan(fj), use_container_width=True)
                st.markdown(f"벽 **{len(fj['walls'])}**개 · 방 **{len(fj['rooms'])}**개")

            floor_json_list.append(fj)

    st.divider()
    col1, _, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("→ Step 2: 구조 확인 및 3D 설정", type="primary", use_container_width=True):
            if len(floor_json_list) > 1:
                st.session_state.floor_json_list = floor_json_list
            else:
                st.session_state.floor_json_list = None
            st.session_state.floor_json = floor_json_list[0]
            st.session_state.step = 2
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# STEP 2: 구조 확인 및 3D 파라미터 설정
# ════════════════════════════════════════════════════════════════════
def step2_confirm():
    st.header("🏗️ Step 2: 구조 확인 및 3D 설정")

    # 다층 or 단층 판별
    fj_list: list[dict] | None = st.session_state.get("floor_json_list")
    is_multi = fj_list is not None and len(fj_list) > 1

    if is_multi:
        _step2_multi(fj_list)
    else:
        _step2_single(st.session_state.floor_json)


def _step2_single(floor_json: dict):
    """단층 도면 확인 및 3D 설정"""
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("건물 구조 편집")
        name = st.text_input("건물 이름", value=floor_json.get("name", "Building"))
        w_m = st.number_input("폭 (m)", 1.0, 200.0, float(floor_json["width"]), 0.5)
        h_m = st.number_input("깊이 (m)", 1.0, 200.0, float(floor_json["height"]), 0.5)

        st.divider()
        st.subheader("3D 파라미터")
        num_floors = st.number_input("층 수 (복제)", 1, 10, 5,
                                     help="같은 평면도를 이 층 수만큼 쌓습니다")
        floor_height = st.slider("층 높이 (m)", 2.4, 5.0, 4.5, 0.1)
        floor_material = st.selectbox("슬래브 재질", ["concrete", "wood", "light"])

        st.divider()
        walls = floor_json.get("walls", [])
        if walls:
            wall_df = pd.DataFrame([{
                "ID": w["id"],
                "시작 (m)": f"({w['start'][0]:.1f},{w['start'][1]:.1f})",
                "끝 (m)": f"({w['end'][0]:.1f},{w['end'][1]:.1f})",
                "재질": w["material"],
            } for w in walls])
            st.dataframe(wall_df, use_container_width=True, height=200)

        with st.expander("⚙️ JSON 직접 편집"):
            edited = st.text_area("JSON", json.dumps(floor_json, indent=2), height=260)
            if st.button("JSON 적용"):
                try:
                    st.session_state.floor_json = json.loads(edited)
                    st.rerun()
                except Exception as e:
                    st.error(f"JSON 오류: {e}")

    with col_right:
        updated_json = dict(floor_json)
        updated_json.update({"name": name, "width": w_m, "height": h_m})
        st.subheader("평면도 미리보기")
        st.plotly_chart(_render_2d_floorplan(updated_json), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("벽", len(floor_json.get("walls", [])))
        c2.metric("방", len(floor_json.get("rooms", [])))
        c3.metric("면적", f"{w_m * h_m:.0f} m²")

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("← 이전"):
            st.session_state.step = 1; st.rerun()
    with col_next:
        if st.button("3D 모델 생성 →", type="primary"):
            st.session_state.floor_json = updated_json
            with st.spinner("3D 모델 생성 중..."):
                try:
                    env = _json_to_env(updated_json)
                    building = build_from_single_env(
                        env, num_floors=int(num_floors),
                        floor_height=floor_height, floor_material=floor_material,
                    )
                    st.session_state.building = building
                    st.session_state.step = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"3D 모델 생성 실패: {e}")


def _step2_multi(fj_list: list[dict]):
    """다층 도면 확인 및 3D 설정"""
    st.info(f"**{len(fj_list)}개 층** 평면도가 검출되었습니다. 각 층을 탭으로 확인하세요.")

    # 공통 3D 파라미터
    col_param, col_info = st.columns([1, 2])
    with col_param:
        st.subheader("3D 파라미터")
        floor_height = st.slider("층 높이 (m)", 2.4, 4.5, 3.0, 0.1)
        floor_material = st.selectbox("슬래브 재질", ["concrete", "wood", "light"])
        bldg_name = st.text_input("건물 이름", "Parsed Building")

    with col_info:
        st.subheader("층별 요약")
        summary_df = pd.DataFrame([{
            "층": f"Floor {i+1}",
            "폭 (m)": round(fj["width"], 1),
            "깊이 (m)": round(fj["height"], 1),
            "벽 수": len(fj["walls"]),
            "방 수": len(fj["rooms"]),
        } for i, fj in enumerate(fj_list)])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 층별 탭 미리보기
    st.divider()
    tab_labels = [f"Floor {i+1}" for i in range(len(fj_list))]
    tabs = st.tabs(tab_labels)
    updated_list = []

    for i, (tab, fj) in enumerate(zip(tabs, fj_list)):
        with tab:
            col_l, col_r = st.columns([1, 1.5])
            with col_l:
                n = st.text_input("층 이름", fj.get("name", f"Floor {i+1}"), key=f"fname_{i}")
                w = st.number_input("폭 (m)", 1.0, 200.0, float(fj["width"]), 0.5, key=f"fw_{i}")
                h = st.number_input("깊이 (m)", 1.0, 200.0, float(fj["height"]), 0.5, key=f"fh_{i}")
                walls = fj.get("walls", [])
                st.markdown(f"벽 **{len(walls)}**개 · 방 **{len(fj.get('rooms',[]))}**개")
                if walls:
                    wall_df = pd.DataFrame([{
                        "ID": w_["id"],
                        "재질": w_["material"],
                        "길이(m)": round(((w_["end"][0]-w_["start"][0])**2+(w_["end"][1]-w_["start"][1])**2)**0.5, 1),
                    } for w_ in walls])
                    st.dataframe(wall_df, use_container_width=True, height=180)
            with col_r:
                updated_fj = dict(fj)
                updated_fj.update({"name": n, "width": w, "height": h})
                st.plotly_chart(_render_2d_floorplan(updated_fj), use_container_width=True)
            updated_list.append(updated_fj)

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("← 이전"):
            st.session_state.step = 1; st.rerun()
    with col_next:
        if st.button(f"3D 모델 생성 ({len(fj_list)}층) →", type="primary"):
            with st.spinner("3D 모델 생성 중..."):
                try:
                    envs = [_json_to_env(fj) for fj in updated_list]
                    building = build_from_env_list(
                        envs, floor_height=floor_height,
                        floor_material=floor_material,
                    )
                    building.name = bldg_name
                    st.session_state.building = building
                    st.session_state.floor_json_list = updated_list
                    st.session_state.step = 3
                    st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"3D 모델 생성 실패: {e}")
                    st.code(traceback.format_exc())


def _json_to_env(floor_json: dict) -> BuildingEnvironment:
    """dict → 임시 JSON 파일 경유 → BuildingEnvironment"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(floor_json, f, ensure_ascii=False)
        tmp = f.name
    try:
        return BuildingEnvironment.from_json(tmp)
    finally:
        import os; os.unlink(tmp)


# ════════════════════════════════════════════════════════════════════
# STEP 3: 3D 모델 미리보기 및 확인
# ════════════════════════════════════════════════════════════════════
def step3_model():
    st.header("🏢 Step 3: 3D 건물 모델 확인")
    building: Building3D = st.session_state.building

    col_info, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("표시 옵션")
        show_outer_only = st.checkbox("외벽만 표시", value=False)
        show_floor_plates = st.checkbox("바닥 슬래브 표시", value=True)
        st.divider()
        st.metric("층 수", building.num_floors)
        st.metric("총 높이", f"{building.total_height_m:.1f} m")
        st.metric("연면적", f"{building.footprint_width * building.footprint_height * building.num_floors:.0f} m²")
        st.metric("벽 수 (층당)", len(building.floors[0].environment.walls))

    with col_info:
        fig3d = build_model_figure(
            building,
            title=f"3D 모델 — {building.name}",
            show_outer_only=show_outer_only,
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # 층 정보 테이블
    st.subheader("층 구성")
    floor_df = pd.DataFrame([
        {
            "층": fl.label,
            "바닥 높이 (m)": f"{fl.z_bottom:.1f}",
            "천장 높이 (m)": f"{fl.z_top:.1f}",
            "층고 (m)": f"{fl.height:.1f}",
            "벽 수": len(fl.environment.walls),
            "방 수": len(fl.environment.rooms),
        }
        for fl in building.floors
    ])
    st.dataframe(floor_df, use_container_width=True)

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("← 이전"):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button("✅ 확인 — AP 배치로 이동", type="primary"):
            st.session_state.step = 4
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# STEP 4: AP 배치 + 시뮬레이션 + 결과
# ════════════════════════════════════════════════════════════════════
def step4_simulate():
    st.header("📡 Step 4: AP 배치 및 시뮬레이션")
    building: Building3D = st.session_state.building
    w = building.footprint_width
    h = building.footprint_height
    floor_options = [fl.label for fl in building.floors]

    # ── 사이드바: 전파 모델 설정 ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 전파 모델 설정")
        model_choice = st.selectbox("전파 모델", ["log_distance", "fspl", "itu_r"])
        pl_exp = st.slider("경로 손실 지수 (n)", 2.0, 5.0, 3.0, 0.1)
        threshold_dbm = st.slider("커버리지 임계값 (dBm)", -100, -40, -75, 1)
        resolution = st.select_slider(
            "격자 해상도 (m)",
            options=[0.25, 0.5, 1.0, 2.0],
            value=0.5,
        )
        noise_floor = st.slider("잡음 바닥 (dBm)", -110, -80, -100, 1)
        st.divider()
        if st.button("← 이전 (3D 모델)", use_container_width=True):
            st.session_state.step = 3
            st.session_state.sim_done = False
            st.rerun()

    # ── 메인: AP 배치 설정 ────────────────────────────────────────────
    col_ap, col_preview = st.columns([1, 1.6])

    with col_ap:
        st.subheader("📡 AP 설정")
        band = st.selectbox("주파수 대역", ["2.4GHz", "5GHz", "6GHz"])
        num_aps = st.number_input("AP 수", 1, 8, 2, key="num_aps_input")

        routers: list[Router] = []
        for i in range(int(num_aps)):
            with st.expander(f"AP {i + 1}", expanded=(i == 0)):
                ap_x = st.slider(
                    f"X 위치 (m)", 0.0, float(w),
                    round(w * (i + 1) / (int(num_aps) + 1), 1), 0.5,
                    key=f"ap_x_{i}",
                )
                ap_y = st.slider(
                    f"Y 위치 (m)", 0.0, float(h),
                    round(h / 2, 1), 0.5,
                    key=f"ap_y_{i}",
                )
                ap_floor_label = st.selectbox(
                    "설치 층", floor_options, index=0, key=f"ap_floor_{i}"
                )
                ap_floor_idx = floor_options.index(ap_floor_label)
                ap_fl = building.floors[ap_floor_idx]
                inner_h_max = max(ap_fl.height - 0.1, 0.6)
                ap_inner_h = st.slider(
                    "층 내 높이 (m)", 0.5, inner_h_max,
                    min(1.5, inner_h_max), 0.1,
                    key=f"ap_innerh_{i}",
                )
                ap_height = ap_fl.z_bottom + ap_inner_h
                ap_tx = st.slider("TX 전력 (dBm)", 10, 30, 20, 1, key=f"ap_tx_{i}")
                max_ch = 13 if band == "2.4GHz" else 196
                ap_ch = st.number_input(
                    "채널", 1, max_ch, min(1 + i * 5, max_ch), key=f"ap_ch_{i}"
                )
                ap_en = st.checkbox("활성화", value=True, key=f"ap_en_{i}")

                routers.append(Router(
                    x=ap_x, y=ap_y,
                    band=band,
                    tx_power_dbm=float(ap_tx),
                    channel=int(ap_ch),
                    ssid=f"AP-{i + 1}",
                    height_m=ap_height,
                    enabled=ap_en,
                ))

    with col_preview:
        st.subheader("AP 배치 미리보기")
        fig_preview = build_model_figure(
            building, routers, title="AP 위치 — 3D 미리보기"
        )
        fig_preview.update_layout(height=420)
        st.plotly_chart(fig_preview, use_container_width=True)

    # ── 시뮬레이션 실행 버튼 (메인 영역 중앙에 배치) ─────────────────
    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_btn = st.button(
            "🚀 신호 강도 맵 생성",
            type="primary",
            use_container_width=True,
        )

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────
    if run_btn:
        # 현재 AP 구성을 세션에 보존
        st.session_state.routers = routers
        st.session_state.sim_params = {
            "model": model_choice,
            "pl_exp": pl_exp,
            "threshold_dbm": threshold_dbm,
            "resolution": resolution,
            "noise_floor": noise_floor,
        }

        params = PropagationParams(
            model=PropagationModel(model_choice),
            path_loss_exponent=pl_exp,
            reference_distance=1.0,
        )
        xs, ys = _build_grid(w, h, resolution)

        with st.spinner("📡 신호 맵 계산 중..."):
            try:
                model3d = PathLossModel3D(building, params, noise_floor_dbm=float(noise_floor))
                floor_maps = model3d.compute_floor_maps(routers, xs, ys)
                summary = model3d.coverage_summary(floor_maps, threshold_dbm=float(threshold_dbm))
                st.session_state.floor_maps = floor_maps
                st.session_state.coverage_summary = summary
                st.session_state.sim_done = True
            except Exception as e:
                st.error(f"시뮬레이션 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        # ★ 핵심: 세션 저장 후 강제 리렌더 → 결과 섹션이 표시됨
        st.rerun()

    # ── 결과 표시 ────────────────────────────────────────────────────
    if not (st.session_state.get("sim_done") and st.session_state.get("floor_maps")):
        st.info("AP 위치를 설정하고 **신호 강도 맵 생성** 버튼을 눌러주세요.")
        return

    floor_maps = st.session_state.floor_maps
    routers_saved = st.session_state.routers
    summary = st.session_state.coverage_summary
    saved_params = st.session_state.get("sim_params", {})
    saved_threshold = saved_params.get("threshold_dbm", threshold_dbm)

    st.success("✅ 시뮬레이션 완료!")

    # KPI 카드
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 커버리지", f"{summary['total_coverage_pct']:.1f}%",
              help=f"≥ {saved_threshold} dBm")
    c2.metric("평균 RSSI", f"{summary['mean_rssi_dbm']:.1f} dBm")
    c3.metric("평균 SINR", f"{summary['mean_sinr_db']:.1f} dB")
    c4.metric("분석 층 수", summary["num_floors"])

    # 층별 요약 테이블
    pf = summary["per_floor"]
    floor_table = pd.DataFrame([
        {
            "층": lbl,
            "커버리지 (%)": f"{v['coverage_pct']:.1f}",
            "평균 RSSI (dBm)": f"{v['mean_rssi']:.1f}",
        }
        for lbl, v in pf.items()
    ])
    st.dataframe(floor_table, use_container_width=True, hide_index=True)

    st.divider()

    # 결과 탭
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ 층별 히트맵", "📦 3D 통합 뷰", "📶 SINR / 용량", "📈 분포 통계"
    ])

    with tab1:
        fig_floors = plot_floor_slices_plotly(
            floor_maps, building, routers_saved,
            map_type="rssi", threshold_dbm=float(saved_threshold),
        )
        st.plotly_chart(fig_floors, use_container_width=True)

    with tab2:
        signal_by_floor = {fm.floor_number: fm.rssi_grid for fm in floor_maps}
        xs_s, ys_s = floor_maps[0].xs, floor_maps[0].ys
        fig_3d = build_model_with_signal_preview(
            building, routers_saved, signal_by_floor, xs_s, ys_s
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        fig_stack = plot_3d_stacked(floor_maps, building, routers_saved)
        st.plotly_chart(fig_stack, use_container_width=True)

    with tab3:
        col_sinr, col_cap = st.columns(2)
        with col_sinr:
            st.plotly_chart(
                plot_floor_slices_plotly(floor_maps, building, routers_saved, map_type="sinr"),
                use_container_width=True,
            )
        with col_cap:
            st.plotly_chart(
                plot_floor_slices_plotly(floor_maps, building, routers_saved, map_type="capacity"),
                use_container_width=True,
            )

    with tab4:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_bar = plot_floor_coverage_bar(floor_maps, threshold_dbm=float(saved_threshold))
        st.pyplot(fig_bar)
        plt.close("all")

        all_rssi = np.concatenate([fm.rssi_grid.flatten() for fm in floor_maps])
        import plotly.express as px
        fig_hist = px.histogram(
            x=all_rssi, nbins=60,
            title="전체 RSSI 분포",
            labels={"x": "RSSI (dBm)", "y": "셀 수"},
            color_discrete_sequence=["#4da6ff"],
            template="plotly_dark",
        )
        fig_hist.add_vline(
            x=saved_threshold, line_dash="dash", line_color="red",
            annotation_text=f"임계값 {saved_threshold} dBm",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    _, col_next5, _ = st.columns([1, 2, 1])
    with col_next5:
        if st.button("📊 Step 5: 빅데이터 최적 배치 분석 →", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# STEP 5: 빅데이터 기반 최소 AP 배치 분석
# ════════════════════════════════════════════════════════════════════
def step5_bigdata():
    st.header("📊 Step 5: 빅데이터 최적 배치 분석")
    st.caption(
        "AP 개수별로 수십~수백 개의 무작위 배치 후보를 대량 시뮬레이션(Monte Carlo)하여 "
        "커버리지/SINR 통계를 분석하고, 목표 성능을 만족하는 **최소 AP 개수**와 "
        "그 중 최고 성능 배치를 추천합니다."
    )

    building: Building3D = st.session_state.building
    w, h = building.footprint_width, building.footprint_height

    col_cfg, col_info = st.columns([1, 1.5])
    with col_cfg:
        st.subheader("⚙️ 분석 설정")
        band = st.selectbox("주파수 대역", ["2.4GHz", "5GHz", "6GHz"], key="bd_band")
        max_routers = st.slider("최대 AP 개수", 1, 6, 4, key="bd_max_routers")
        samples_per_count = st.slider(
            "AP 개수별 샘플 수 (빅데이터 규모)", 5, 100, 20, 5, key="bd_samples",
            help="값이 클수록 더 많은 무작위 배치를 시뮬레이션해 통계 신뢰도가 높아지지만 시간이 오래 걸립니다.",
        )
        threshold_dbm = st.slider("커버리지 임계값 (dBm)", -100, -40, -75, 1, key="bd_threshold")
        target_coverage_pct = st.slider("목표 커버리지 (%)", 10, 100, 70, 5, key="bd_target")
        resolution = st.select_slider(
            "분석 격자 해상도 (m)", options=[1.0, 2.0], value=1.0, key="bd_resolution",
            help="해상도가 낮을수록(값이 클수록) 분석 속도가 빠릅니다.",
        )
        model_choice = st.selectbox("전파 모델", ["log_distance", "fspl", "itu_r"], key="bd_model")
        pl_exp = st.slider("경로 손실 지수 (n)", 2.0, 5.0, 3.0, 0.1, key="bd_pl_exp")
        seed = st.number_input("랜덤 시드", 0, 99999, 42, key="bd_seed")

        total_eval = max_routers * samples_per_count
        st.info(f"총 **{total_eval:,}개** 배치 후보를 시뮬레이션합니다 (AP 1~{max_routers}개 × {samples_per_count}샘플).")

        run_btn = st.button("🔍 빅데이터 분석 시작", type="primary", use_container_width=True)

    if run_btn:
        params = PropagationParams(
            model=PropagationModel(model_choice),
            path_loss_exponent=pl_exp,
            reference_distance=1.0,
        )
        xs, ys = _build_grid(w, h, resolution)

        progress = st.progress(0, text="빅데이터 시뮬레이션 준비 중...")

        def _cb(done, total):
            progress.progress(done / total, text=f"배치 후보 시뮬레이션 중... {done}/{total}")

        try:
            result = run_big_data_search(
                building, params, band, xs, ys,
                threshold_dbm=float(threshold_dbm),
                target_coverage_pct=float(target_coverage_pct),
                max_routers=int(max_routers),
                samples_per_count=int(samples_per_count),
                seed=int(seed),
                progress_callback=_cb,
            )
            progress.progress(1.0, text="완료!")
            st.session_state.bigdata_result = result
            st.session_state.bigdata_band = band
        except Exception as e:
            import traceback
            st.error(f"빅데이터 분석 오류: {e}")
            st.code(traceback.format_exc())
            return

    with col_info:
        result = st.session_state.get("bigdata_result")
        if result is None:
            st.info("**빅데이터 분석 시작** 버튼을 눌러주세요.")
        else:
            st.subheader("📈 빅데이터 분석 규모")
            cells_per_eval = result.total_data_points // result.total_samples
            m1, m2, m3 = st.columns(3)
            m1.metric("시뮬레이션 배치 후보", f"{result.total_samples:,}개")
            m2.metric("배치당 분석 셀 수", f"{cells_per_eval:,}개")
            m3.metric("총 분석 데이터 포인트", f"{result.total_data_points:,}개")

            st.divider()
            st.subheader("🏆 추천 결과")
            if result.recommended_num_aps is not None:
                n = result.recommended_num_aps
                st.success(
                    f"✅ **최소 {n}개의 AP**로 목표 커버리지 "
                    f"{result.target_coverage_pct:.0f}% (≥ {result.threshold_dbm:.0f} dBm) 달성 가능"
                )
            else:
                st.warning(
                    f"⚠️ AP {max_routers}개까지 시뮬레이션했지만 목표 커버리지 "
                    f"{result.target_coverage_pct:.0f}%를 만족하는 배치를 찾지 못했습니다. "
                    "최대 AP 개수를 늘리거나 목표 커버리지를 낮춰보세요."
                )

    if result is None:
        return

    st.divider()

    # ── AP 개수별 통계 테이블 ────────────────────────────────────────
    st.subheader("📊 AP 개수별 커버리지 통계")
    stats_rows = [cs.stats() for cs in result.by_count.values()]
    stats_df = pd.DataFrame([
        {
            "AP 수": s["num_aps"],
            "샘플 수": s["n_samples"],
            "평균 커버리지(%)": round(s["mean"], 1),
            "표준편차": round(s["std"], 2),
            "최소(%)": round(s["min"], 1),
            "최대(%)": round(s["max"], 1),
            "중앙값(%)": round(s["median"], 1),
            "90백분위(%)": round(s["p90"], 1),
            "최고 평균 RSSI(dBm)": round(s["best_mean_rssi"], 1),
            "최고 평균 SINR(dB)": round(s["best_mean_sinr"], 1),
        }
        for s in stats_rows
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── 분포 박스플롯 ────────────────────────────────────────────────
    import plotly.express as px
    box_data = []
    for cs in result.by_count.values():
        for s in cs.samples:
            box_data.append({"AP 수": cs.num_aps, "커버리지(%)": s.coverage_pct})
    box_df = pd.DataFrame(box_data)

    fig_box = px.box(
        box_df, x="AP 수", y="커버리지(%)", points="all",
        title=f"AP 개수별 커버리지 분포 ({result.total_samples}개 배치 후보, ≥ {result.threshold_dbm:.0f} dBm 기준)",
        template="plotly_dark", color="AP 수",
    )
    fig_box.add_hline(
        y=result.target_coverage_pct, line_dash="dash", line_color="red",
        annotation_text=f"목표 {result.target_coverage_pct:.0f}%",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── 추천 배치 상세 ───────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 최적 배치 상세")

    best_n = result.recommended_num_aps or max(result.by_count.keys())
    best_sample = result.by_count[best_n].best
    band = st.session_state.get("bigdata_band", "5GHz")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AP 개수", best_n)
    c2.metric("커버리지", f"{best_sample.coverage_pct:.1f}%")
    c3.metric("평균 RSSI", f"{best_sample.mean_rssi:.1f} dBm")
    c4.metric("평균 SINR", f"{best_sample.mean_sinr:.1f} dB")

    routers_df = pd.DataFrame([
        {
            "AP": f"AP-{i+1}",
            "X (m)": round(r.x, 1),
            "Y (m)": round(r.y, 1),
            "높이 (m)": round(r.height_m, 1),
            "채널": r.channel,
            "TX (dBm)": round(r.tx_power_dbm, 1),
        }
        for i, r in enumerate(best_sample.routers)
    ])
    st.dataframe(routers_df, use_container_width=True, hide_index=True)

    fig_best = build_model_figure(
        building, best_sample.routers,
        title=f"추천 배치 — AP {best_n}개 (커버리지 {best_sample.coverage_pct:.1f}%)",
    )
    fig_best.update_layout(height=500)
    st.plotly_chart(fig_best, use_container_width=True)

    st.divider()
    if st.button("← Step 4로 이동"):
        st.session_state.step = 4
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# 2D 평면도 렌더링 (Step 2 미리보기)
# ════════════════════════════════════════════════════════════════════
def _render_2d_floorplan(floor_json: dict) -> go.Figure:
    fig = go.Figure()
    walls = floor_json.get("walls", [])
    rooms = floor_json.get("rooms", [])
    w = floor_json.get("width", 20)
    h = floor_json.get("height", 15)

    # 배경 사각형
    fig.add_shape(type="rect", x0=0, y0=0, x1=w, y1=h,
                  fillcolor="rgba(30,30,50,0.3)", line=dict(color="rgba(100,100,100,0.5)", width=1))

    # 방 배경
    for r in rooms:
        b = r.get("bounds", [0, 0, 1, 1])
        fig.add_shape(
            type="rect", x0=b[0], y0=b[1], x1=b[2], y1=b[3],
            fillcolor="rgba(70,100,150,0.12)",
            line=dict(color="rgba(100,150,200,0.3)", width=1),
        )
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        fig.add_annotation(
            x=cx, y=cy, text=r.get("name", ""),
            showarrow=False, font=dict(size=9, color="rgba(180,200,255,0.7)"),
        )

    # 벽
    for wall in walls:
        mat = wall.get("material", "drywall")
        if mat == "concrete":
            color, lw = "rgba(180,170,160,0.95)", 4
        elif mat == "brick":
            color, lw = "rgba(160,82,45,0.9)", 3
        elif mat == "glass":
            color, lw = "rgba(135,206,235,0.7)", 2
        elif "door" in mat:
            color, lw = "rgba(139,69,19,0.8)", 2
        else:
            color, lw = "rgba(140,140,160,0.80)", 2

        fig.add_trace(go.Scatter(
            x=[wall["start"][0], wall["end"][0]],
            y=[wall["start"][1], wall["end"][1]],
            mode="lines",
            line=dict(color=color, width=lw),
            showlegend=False,
            hovertemplate=f"{wall['id']}<br>{mat}<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(title="X (m)", range=[-0.5, w + 0.5], scaleanchor="y"),
        yaxis=dict(title="Y (m)", range=[-0.5, h + 0.5], autorange="reversed"),
        template="plotly_dark",
        height=350,
        margin=dict(l=30, r=10, t=20, b=30),
        plot_bgcolor="rgb(15,15,25)",
        paper_bgcolor="rgb(15,15,25)",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# 메인 진입점
# ════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Wi-Fi 3D Simulator",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("📡 Wi-Fi 3D 신호 강도 시뮬레이터")

    _init_state()

    step = st.session_state.step
    _step_indicator(step)

    if step == 1:
        step1_upload()
    elif step == 2:
        step2_confirm()
    elif step == 3:
        step3_model()
    elif step == 4:
        step4_simulate()
    elif step == 5:
        step5_bigdata()


if __name__ == "__main__":
    main()
