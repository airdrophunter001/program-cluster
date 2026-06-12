"""
Wi-Fi 공유기(AP) 객체 정의
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid


BAND_DEFAULTS = {
    "2.4GHz": {"frequency_hz": 2.4e9, "channel_bandwidth_mhz": 20, "tx_power_dbm": 20},
    "5GHz":   {"frequency_hz": 5.0e9, "channel_bandwidth_mhz": 80, "tx_power_dbm": 23},
    "6GHz":   {"frequency_hz": 6.0e9, "channel_bandwidth_mhz": 160, "tx_power_dbm": 23},
}


@dataclass
class Router:
    x: float
    y: float
    band: str = "2.4GHz"
    tx_power_dbm: float = 20.0
    antenna_gain_dbi: float = 2.0
    rx_sensitivity_dbm: float = -90.0
    channel: int = 6
    ssid: str = ""
    router_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    height_m: float = 1.5  # 안테나 높이 (바닥 기준)
    enabled: bool = True

    def __post_init__(self):
        if not self.ssid:
            self.ssid = f"AP_{self.router_id}"
        if self.band not in BAND_DEFAULTS:
            raise ValueError(f"Unsupported band: {self.band}. Use one of {list(BAND_DEFAULTS)}")

    @property
    def frequency_hz(self) -> float:
        return BAND_DEFAULTS[self.band]["frequency_hz"]

    @property
    def channel_bandwidth_mhz(self) -> float:
        return BAND_DEFAULTS[self.band]["channel_bandwidth_mhz"]

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def effective_tx_power_dbm(self) -> float:
        return self.tx_power_dbm + self.antenna_gain_dbi

    def distance_to(self, x: float, y: float) -> float:
        import math
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def to_dict(self) -> dict:
        return {
            "id": self.router_id,
            "ssid": self.ssid,
            "x": self.x,
            "y": self.y,
            "band": self.band,
            "tx_power_dbm": self.tx_power_dbm,
            "antenna_gain_dbi": self.antenna_gain_dbi,
            "channel": self.channel,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Router":
        return cls(
            x=data["x"],
            y=data["y"],
            band=data.get("band", "2.4GHz"),
            tx_power_dbm=data.get("tx_power_dbm", 20.0),
            antenna_gain_dbi=data.get("antenna_gain_dbi", 2.0),
            channel=data.get("channel", 6),
            ssid=data.get("ssid", ""),
            router_id=data.get("id", ""),
        )


def check_channel_interference(r1: Router, r2: Router) -> float:
    """
    두 AP 간 채널 간섭 계수 반환 (0=무간섭, 1=완전 간섭)
    2.4GHz: 채널 간격 5MHz, 비중첩 채널 1/6/11
    5GHz/6GHz: 채널이 겹치지 않으면 0
    """
    if r1.band != r2.band:
        return 0.0

    if r1.band == "2.4GHz":
        ch_diff = abs(r1.channel - r2.channel)
        if ch_diff == 0:
            return 1.0
        elif ch_diff < 5:
            return max(0.0, 1.0 - ch_diff / 5.0)
        else:
            return 0.0
    else:
        return 1.0 if r1.channel == r2.channel else 0.0


def create_default_routers(band: str = "2.4GHz") -> list[Router]:
    """기본 공유기 배치 예시"""
    defaults = [
        Router(x=4.0,  y=7.5,  band=band, channel=1,  ssid="AP-Left"),
        Router(x=11.0, y=3.5,  band=band, channel=6,  ssid="AP-Center"),
        Router(x=17.0, y=10.0, band=band, channel=11, ssid="AP-Right"),
    ]
    return defaults
