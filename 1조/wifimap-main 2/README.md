# Wi-Fi Heatmap Simulator

건물 평면도 이미지를 업로드하면 3D 모델로 변환하고 Wi-Fi 신호 강도 히트맵을 시뮬레이션합니다.

## 실행 방법

### 1. Docker (팀 공유 / 서버 배포)

```bash
# 빌드 & 실행
docker compose up -d

# 브라우저에서 접속
http://localhost:8501
```

서버에 올릴 경우 `localhost` 대신 서버 IP로 접속합니다.

```bash
# 중지
docker compose down

# 로그 확인
docker compose logs -f
```

---

### 2. Streamlit Cloud (데모 / 외부 공유)

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub 계정 연결
3. `me8114/wifimap` 레포 선택
4. Main file path: `wifi_heatmap/visualization/interactive.py`
5. Deploy 클릭

배포 후 생성된 URL을 팀에 공유하면 누구나 브라우저에서 바로 사용 가능합니다.

---

### 3. 로컬 실행

```bash
pip install -r requirements.txt
streamlit run wifi_heatmap/visualization/interactive.py
```

## 주요 기능

- 평면도 이미지 → 자동 벽 감지 → 3D 건물 모델 변환
- 다층 건물: 층별 다른 평면도 자동 분리 인식
- Wi-Fi 신호 (RSSI / SINR / 용량) 히트맵
- AP 배치 유전 알고리즘 최적화
- CSV / PDF 리포트 내보내기
