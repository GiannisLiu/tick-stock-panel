"""通达信(本机客户端 17709 端口)数据源 provider。

实现数据集:
  - depth5      五档盘口, 逐标的 get_market_snapshot; 分片/限速由 depth_service 按
                depth5.batch(rpm) 统一负责, 本 provider 不自行分片或回退其它源
  - full_minute 盘中全市场当日 1 分钟K(修复轮 get_intraday_batch + 增量轮
                get_intraday_latest), 节奏由 minute_refresh 调度

未声明 realtime → 通达信只有单标的快照接口, 全市场 5575 只 ≈ 72s/轮(实测 12.7ms/次),
  与 6s 轮询差一个数量级, 声明了只会拖垮行情轮询, 自动回退 TickFlow。
未声明 minute → 分时图/分钟回测已有 stocksdk 浅源与本插件解耦, 不重复接管。

单位与口径 (CONTRIBUTING §3.1, 不可凭字段名推断):
  - 快照 Volume 单位为手, 与盘口契约一致(Buyp/Buyv 等同批口径), 直接透传;
    而 K 线 Volume 单位是股 → 分钟帧必须 /100 成手。同一个客户端两套口径, 混用
    会得到看似合理的 100 倍错误。
  - K 线 Amount 单位为万元 → 分钟帧契约要元, 必须 x10000。
  - 通达信对空档位返回 "0.00"/"0", 映射为 0.0/0 而不是 None: depth_service 用
    ask1 == 0 判定涨停真封、bid1 == 0 判定跌停真封, None 会退化成"无法判定"。
  - 快照无服务端时间戳字段 → timestamp 取本轮本地墙钟毫秒, 一轮内共用同一值。
  - 分钟帧 datetime 由 Date(20260921) + Time(93100) 拼成北京墙钟 naive(9:31-15:00,
    落在守卫的北京特征时段内直通)。

全量分钟的冷启动语义(实测 2026-09): 客户端首次取某只标的 1 分钟线时是**按需从服务器
补**, 每一遍约往前推进 4 个交易日, 全市场 5575 只在 3 遍(~70s)内达到当日覆盖 99.8%。
所以 provider 坚持"只输出当日 bar": 冷缓存时旧日期 bar 被丢弃 → 空轮 →
minute_refresh 自愈触发修复轮, 数十秒内向当日收敛; 反之若把旧日期 bar 当当日写进
存储, 会静默污染分钟库(不报错、页面照渲染)。

后端跑在 Docker 里时(本项目默认部署方式)容器的 loopback 不是宿主机, 需用环境变量把地址
指向宿主机: TDX_BASE_URL=http://host.docker.internal:17709/, 并把插件目录挂进容器。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import polars as pl

from app.market_time import cn_today
from app.plugins.tdx import client as tdx_client
from app.plugins.tdx.client import PROBE_SYMBOL, TdxError

logger = logging.getLogger(__name__)

_LEVELS = 5
# get_market_data 单请求标的数上限: 超出部分静默丢弃(传 1500 只也只回 100)
_BATCH = 100
# A 股每交易日 1 分钟根数(9:31-15:00)。当日最多这么多, 多要只会拉回前一日尾巴。
_DAY_MINUTE_BARS = 240
# 只声明真实提供的数据集; 其余数据集 provider_has_dataset 为 False → 回退 TickFlow
_DATASETS = {"depth5": True, "full_minute": True}

_MINUTE_SCHEMA = {
    "symbol": pl.Utf8,
    "datetime": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
}


class _TdxConfig:
    """dataset 声明: services 层靠 provider_has_dataset 路由。"""

    datasets = _DATASETS


def _num(raw: object) -> float:
    """通达信字段是字符串("1252.57"), 也兼容数字; 无法解析归 0。"""
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _pad(raw: object) -> list:
    """取前五档并补位: 通达信可能缺尾(返回 1~5 项), 或整体缺失。"""
    if not isinstance(raw, list):
        return ["0"] * _LEVELS
    items = list(raw[:_LEVELS])
    return items + ["0"] * (_LEVELS - len(items))


def _map_depth(row: dict, ts_ms: int) -> dict | None:
    """快照行 → 盘口契约字典; 五档全空返回 None(不产出条目)。"""
    bid_prices = [_num(x) for x in _pad(row.get("Buyp"))]
    bid_volumes = [int(_num(x)) for x in _pad(row.get("Buyv"))]
    ask_prices = [_num(x) for x in _pad(row.get("Sellp"))]
    ask_volumes = [int(_num(x)) for x in _pad(row.get("Sellv"))]
    if not any(bid_prices + bid_volumes + ask_prices + ask_volumes):
        # 停牌/未订阅等无盘口场景: 留空比给一排 0 安全 —— 后者会被判成"涨停真封"
        return None
    return {
        "bid_prices": bid_prices,
        "bid_volumes": bid_volumes,
        "ask_prices": ask_prices,
        "ask_volumes": ask_volumes,
        "timestamp": ts_ms,
    }


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _minute_rows(symbol: str, row: dict, day: str) -> list[tuple]:
    """一只标的的原始列数组 → canonical 8 列行, 只保留 day(YYYYMMDD) 当日的 bar。

    单位: Volume 股 → 手(÷100), Amount 万元 → 元(x10000); Amount 缺失置 None。
    """
    dates = row.get("Date") or []
    times = row.get("Time") or []
    opens = row.get("Open") or []
    highs = row.get("High") or []
    lows = row.get("Low") or []
    closes = row.get("Close") or []
    volumes = row.get("Volume") or []
    amounts = row.get("Amount") or []
    n = min(len(dates), len(times), len(opens), len(highs), len(lows), len(closes), len(volumes))
    out: list[tuple] = []
    for i in range(n):
        if dates[i] != day:
            continue  # 冷缓存时回的是历史 bar, 当当日写进去会静默污染分钟库
        ts = str(times[i]).zfill(6)
        try:
            dt = datetime(  # 北京时间墙钟 naive, 与日K的 date 语义对齐
                int(day[0:4]),
                int(day[4:6]),
                int(day[6:8]),
                int(ts[0:2]),
                int(ts[2:4]),
                int(ts[4:6]),
            )
        except ValueError:
            continue  # 时间字段畸形 → 丢弃该根, 不猜
        out.append(
            (
                symbol,
                dt,
                _num(opens[i]),
                _num(highs[i]),
                _num(lows[i]),
                _num(closes[i]),
                _num(volumes[i]) / 100.0,
                _num(amounts[i]) * 10000.0 if i < len(amounts) else None,
            )
        )
    return out


def _minute_frame(rows: list[tuple]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_MINUTE_SCHEMA)  # 空帧也要带列, 让空轮可识别
    return pl.DataFrame(rows, schema=_MINUTE_SCHEMA, orient="row")


class TdxProvider:
    """通达信数据源(五档盘口 + 全量分钟)。"""

    name = "tdx"
    builtin = True

    def __init__(self) -> None:
        self.config = _TdxConfig()
        self._client: tdx_client.TdxClient | None = None
        self._codes: list[str] | None = None
        self._codes_date: object = None

    def close(self) -> None:  # loader.load_all 重建注册表时会对每个 provider 调 close
        if self._client is not None:
            self._client.close()
            self._client = None
        self._codes = None
        self._codes_date = None

    def _get_client(self) -> tdx_client.TdxClient:
        if self._client is None:
            self._client = tdx_client.TdxClient()
        return self._client

    # ---- 数据集实现 ----

    def get_depth_batch(self, symbols: list[str]) -> dict[str, dict]:
        """五档盘口: [symbol] → 标准盘口字典。单标的失败只跳过该标的。"""
        if not symbols:
            return {}
        client = self._get_client()
        ts_ms = int(time.time() * 1000)
        result: dict[str, dict] = {}
        for symbol in symbols:
            try:
                row = client.snapshot(symbol)
            except TdxError as e:
                logger.warning("通达信盘口取数失败 %s: %s", symbol, e)
                continue
            mapped = _map_depth(row, ts_ms)
            if mapped is not None:
                result[symbol] = mapped
        return result

    # ---- full_minute: 盘中全市场当日 1 分钟K ----

    def _market_codes(self) -> list[str]:
        """全市场代码当日缓存: 一个交易日内维表不会变, 不必每轮重拉。"""
        if self._codes is None or self._codes_date != cn_today():
            self._codes = self._get_client().market_codes()
            self._codes_date = cn_today()
        return self._codes

    def _intraday_frame(self, symbols: list[str], count: int) -> pl.DataFrame:
        """当日 1 分钟K: 分块取数(≤100 只/请求), 只保留当日 bar。"""
        if not symbols:
            return _minute_frame([])
        client = self._get_client()
        day = cn_today().strftime("%Y%m%d")
        # 当日最多 240 根; 要更多只会拉回前一日尾巴(随后被过滤掉) → 压到 241 省传输
        want = max(1, min(int(count or 1), _DAY_MINUTE_BARS + 1))
        rows: list[tuple] = []
        for chunk in _chunked(list(symbols), _BATCH):
            try:
                data = client.kline_1m(chunk, want)
            except TdxError as e:
                logger.warning("通达信分钟取数失败(%d 只): %s", len(chunk), e)
                continue
            for symbol, row in data.items():
                rows.extend(_minute_rows(symbol, row, day))
        return _minute_frame(rows)

    def get_intraday_batch(
        self, symbols: list[str], count: int = 300, asset_type: str = "stock"
    ) -> pl.DataFrame:
        """修复轮: 给定标的的当日 1 分钟K(canonical 8 列, 北京墙钟 naive)。"""
        return self._intraday_frame(symbols, count)

    def get_intraday_latest(self, symbols: list[str] | None = None, count: int = 3) -> pl.DataFrame:
        """稳态增量轮: 全市场每只最新 count 根 1 分钟K(56 请求, 实测约 2.9s)。"""
        return self._intraday_frame(symbols if symbols else self._market_codes(), count)

    # ---- 设置页「试拉」 ----

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        if dataset == "full_minute":
            syms = [s for s in (symbols or [])][:3] or [PROBE_SYMBOL]
            try:
                df = self.get_intraday_batch(syms, count=_DAY_MINUTE_BARS)
            except TdxError as e:
                return {"provider": self.name, "dataset": dataset, "rows": 0, "error": str(e)}
            preview = df.head(3).to_dicts()
            for r in preview:  # datetime 要 JSON 可序列化
                r["datetime"] = r["datetime"].isoformat() if r["datetime"] else None
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": df.height,
                "columns": df.columns,
                "preview": preview,
                # 非交易时段/冷缓存时为空是预期行为, 不是错, 但要说清楚
                "error": None if df.height else "当日暂无 1 分钟K(非交易时段, 或冷缓存需多轮取数)",
            }
        if dataset != "depth5":
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": 0,
                "error": f"通达信插件未接入 {dataset} 数据集(自动回退 TickFlow)",
            }
        syms = [s for s in (symbols or [])][:3] or [PROBE_SYMBOL]
        try:  # 先单标的实探一次, 客户端没开时给出明确原因而不是 rows=0
            self._get_client().snapshot(syms[0])
        except TdxError as e:
            return {"provider": self.name, "dataset": dataset, "rows": 0, "error": str(e)}
        data = self.get_depth_batch(syms)
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": len(data),
            "columns": [
                "symbol",
                "bid_prices",
                "bid_volumes",
                "ask_prices",
                "ask_volumes",
                "timestamp",
            ],
            "preview": [{"symbol": s, **d} for s, d in list(data.items())[:3]],
        }


def availability() -> tuple[bool, str]:
    """loader 启动自检: 客户端在运行且行情可用才注册为可切换源。不抛异常。"""
    base_url = tdx_client.default_base_url()
    try:
        ok = tdx_client.TdxClient().ping()
    except Exception as e:  # 自检绝不把异常抛给 loader
        return False, f"通达信服务探测异常({base_url}): {e}"
    if ok:
        return True, "ok"
    docker_hint = (
        "; Docker 部署请设 TDX_BASE_URL=http://host.docker.internal:17709/"
        if "127.0.0.1" in base_url
        else ""
    )
    return (
        False,
        f"通达信客户端未运行或未登录(本插件经 {base_url} 取五档, 需保持客户端开启{docker_hint})",
    )
