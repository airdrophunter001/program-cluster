import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 맵 크기
# ==========================================

WIDTH = 120
HEIGHT = 80

# ==========================================
# 건물 맵
# 0 = 빈 공간
# 1 = 콘크리트
# 2 = 유리
# ==========================================

building = np.zeros((HEIGHT, WIDTH))

# 외벽
building[0, :] = 1
building[-1, :] = 1
building[:, 0] = 1
building[:, -1] = 1

# 내부 벽
building[20:70, 40] = 1
building[10:60, 80] = 2

# ==========================================
# 재질 감쇠
# ==========================================

loss_table = {
    1: 15,  # 콘크리트
    2: 5    # 유리
}

# ==========================================
# Wi-Fi 설정
# ==========================================

P0 = -30
n = 2.2

# ==========================================
# 초기 라우터 위치
# ==========================================

router_x = 20
router_y = 40

# ==========================================
# 벽 감쇠 계산
# ==========================================

def wall_loss(x1, y1, x2, y2):

    total_loss = 0

    samples = 80

    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)

    for i in range(samples):

        xi = int(xs[i])
        yi = int(ys[i])

        material = building[yi, xi]

        if material in loss_table:
            total_loss += loss_table[material]

    return total_loss / 10


# ==========================================
# 신호 계산
# ==========================================

def generate_signal_map(router_x, router_y):

    signal_map = np.zeros((HEIGHT, WIDTH))

    for y in range(HEIGHT):
        for x in range(WIDTH):

            d = np.sqrt((x - router_x)**2 + (y - router_y)**2)

            if d < 1:
                d = 1

            # 거리 감쇠
            rssi = P0 - 10 * n * np.log10(d)

            # 벽 감쇠
            rssi -= wall_loss(router_x, router_y, x, y)

            # ==========================
            # 반사 효과
            # ==========================

            near_wall = False

            for dy in range(-2, 3):
                for dx in range(-2, 3):

                    nx = x + dx
                    ny = y + dy

                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        if building[ny, nx] != 0:
                            near_wall = True

            if near_wall:
                rssi += 2

            # ==========================
            # 회절 효과
            # ==========================

            if rssi < -75:
                rssi += 5

            # ==========================
            # 다중경로 노이즈
            # ==========================

            noise = np.random.normal(0, 1.2)

            rssi += noise

            signal_map[y, x] = rssi

    return signal_map


# ==========================================
# 초기 신호맵
# ==========================================

signal_map = generate_signal_map(router_x, router_y)

# ==========================================
# 그래프 생성
# ==========================================

fig, ax = plt.subplots(figsize=(14, 8))

plt.subplots_adjust(bottom=0.15)

heatmap = ax.imshow(
    signal_map,
    cmap='turbo',
    origin='lower',
    vmin=-100,
    vmax=-30
)

# ==========================================
# 벽 시각화
# ==========================================

wall_y, wall_x = np.where(building == 1)

ax.scatter(
    wall_x,
    wall_y,
    color='black',
    s=10,
    label='Concrete'
)

glass_y, glass_x = np.where(building == 2)

ax.scatter(
    glass_x,
    glass_y,
    color='cyan',
    s=10,
    label='Glass'
)

# ==========================================
# 라우터 표시
# ==========================================

router_plot = ax.scatter(
    router_x,
    router_y,
    color='white',
    s=300,
    marker='*',
    edgecolors='black',
    label='Router'
)

# ==========================================
# 컬러바
# ==========================================

cbar = plt.colorbar(heatmap)

cbar.set_label("Wi-Fi Signal Strength (dBm)")

# ==========================================
# 제목
# ==========================================

ax.set_title("Interactive Wi-Fi Signal Simulation")

ax.legend(loc='upper right')

# ==========================================
# 드래그 기능
# ==========================================

dragging = False

def on_press(event):

    global dragging

    if event.inaxes != ax:
        return

    contains, _ = router_plot.contains(event)

    if contains:
        dragging = True


def on_release(event):

    global dragging

    dragging = False


def on_motion(event):

    global router_x, router_y

    if not dragging:
        return

    if event.inaxes != ax:
        return

    router_x = int(event.xdata)
    router_y = int(event.ydata)

    new_map = generate_signal_map(router_x, router_y)

    heatmap.set_data(new_map)

    router_plot.set_offsets([[router_x, router_y]])

    fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()