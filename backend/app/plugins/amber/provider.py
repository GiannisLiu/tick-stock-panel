"""amber(自部署行情服务)内置数据源 provider。

方法签名对齐 custom.GenericHTTPProvider(service 分流点按这套签名调用),
注入 custom loader 注册表后, 各 service 无需改动即可路由到本 provider。

实现数据集:
  - daily        A 股日K(股票; ETF 代码透传亦可查), 原始价。amber 日K由分钟仓库
                 按日聚合, 历史深度随其仓库积累变厚(新装服务只有近端)。
  - full_minute  全量分钟: get_intraday_batch 全天修复轮(当日窗口批量) +
                 get_intraday_latest 稳态增量轮(全市场每只最新 count 根, 按市场
                 分段单请求)。未声明 minute(个股分时)数据集 → 回退 tickflow。
  - realtime     A 股+ETF 全市场列式快照(GET /v1/snapshot, 单请求一轮)。
                 amber 标的池含 ETF → ETF 实时计数天然非零(优于无 ETF 的源);
                 无指数行情 → 未实现 get_realtime_indices, 指数由本地日K兜底
                 (quote_service 语义, 与 fuyao 的指数降级一致)。
未声明 adj_factor / financial / minute → provider_has_dataset 为 False, 回退 tickflow。

单位与口径 (CONTRIBUTING §3.1, 不可凭字段名推断 — 均经 amber 源码核实):
  - amber volume 单位为手(contracts.py 明示"volume：手"), 与本项目日K/分钟/实时
    契约一致 → 直接透传, 不做 /100 (对比: fuyao 为股需 /100)。
  - amount 单位元, 一致, 直接透传。
  - 快照无涨跌幅字段 → change_pct 由 (last-pre)/pre 推导, 天然小数制, 无 /100。
  - 快照无逐行时间戳 → timestamp 用本地时刻兜底(契约允许的降级路径)。
  - 日K为分钟聚合原始价, 无复权概念 → 契约要求的"不复权原始价"天然满足。
  - 分钟 ts 为北京墙钟(Asia/Shanghai), 契约要求 naive 北京墙钟 → tz-aware 输入
    在 _minute_dt 内归一为 naive; kline_sync 入口守卫为最终兜底。

分块护栏 (amber 服务端): 单请求 codes ≤2000, 分钟响应总 bar ≤200k,
日K响应 cell ≤400k(超限从最早日期截断, 置 truncated)。batch 大小据此计算。
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from app.data_providers.normalizer import normalize_daily
from app.market_time import cn_today
from app.plugins.amber import client as amber_client
from app.plugins.amber.client import (
    MINUTES_CODES_MAX,
    MINUTES_COUNT_MAX,
    AmberError,
)

logger = logging.getLogger(__name__)

# 只声明真实提供的数据集; 其余数据集 provider_has_dataset 返回 False → 回退 tickflow
_DATASETS = ("daily", "full_minute", "realtime")

API_KEY_ENV = "AMBER_API_KEY"
SECRETS_FIELD = "amber_api_key"  # UI 配置的 Key 存 secrets.json, 优先级高于 .env
BASE_URL_ENV = "AMBER_BASE_URL"  # 自部署服务地址, 默认本机 8310

_BJ = ZoneInfo("Asia/Shanghai")
# 分钟单请求 codes 上限: 服务端 bar 总上限 200k ÷ 全日 ~240 根 ≈ 833, 留余量取 700
_MINUTE_CODES_PER_REQ = 700
_INTERVAL_S = 0.1  # 批间隔, 服务端限流默认 120 请求/分钟/Key, 全市场日K ~50 批足够宽裕
_DAILY_CODES_MAX = 1000  # 单请求 cell 上限 400k ÷ 批大小, 按窗口跨度动态收窄
_DAILY_CELL_BUDGET = 300_000  # 400k 上限内留余量, 避免边界触发服务端截断
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


def get_api_key() -> str:
    from app import secrets_store

    return secrets_store.get_env_backed_secret(SECRETS_FIELD, API_KEY_ENV)


def get_base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or "").strip() or amber_client.DEFAULT_BASE_URL


def availability() -> tuple[bool, str]:
    """loader 启动自检: API Key 已配置(secrets.json 或 .env)才注册为可切换数据源。不抛异常。"""
    if get_api_key():
        return True, "ok"
    # 状态行会拼在「未配置」标签之后, 文案不再重复"未配置"字样
    return False, f"缺少 API Key(可在下方输入框直接填写,或配置环境变量 {API_KEY_ENV})"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    """用候选 Key 实探一次标的清单接口(先探后存, 对齐 /tickflow-key 语义)。不落盘。"""
    client = None
    try:
        client = amber_client.AmberClient(api_key=api_key, base_url=get_base_url(), timeout=10.0)
        payload = client.instruments(market="SH")
        n = payload.get("n")
        return True, f"ok (SH 标的 {n} 只)" if n is not None else "ok"
    except AmberError as e:
        return False, f"Key 无效或服务不可达({get_base_url()}): {e}"
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


@dataclass
class _AmberConfig:
    """轻量 config shim, 让 custom loader 的 provider_has_dataset 能识别本 provider。"""

    name: str = "amber"
    display_name: str = "amber"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minute_dt(value) -> datetime | None:
    """amber ts(ISO 串) → naive 北京墙钟 datetime。tz-aware 输入显式归一。

    契约要求 naive 北京墙钟; amber 仓库存储即北京墙钟, 这里只兜底处理
    带偏移的形态(如 "+08:00"), 不做启发式时区猜测(入口守卫兜底)。
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_BJ).replace(tzinfo=None)
    return dt


def _minute_rows(bars: dict) -> list[dict]:
    """amber minutes bars {code: [{ts,o,h,l,c,v,a}]} → canonical 8 列行。

    close/datetime 缺失的行跳过(不伪造); volume 为手、amount 为元, 直接透传。
    """
    rows: list[dict] = []
    for code, bar_list in (bars or {}).items():
        if not isinstance(bar_list, list):
            continue
        for bar in bar_list:
            if not isinstance(bar, dict):
                continue
            close = _to_float(bar.get("c"))
            dt = _minute_dt(bar.get("ts"))
            if close is None or dt is None:
                continue
            rows.append({
                "symbol": str(code),
                "datetime": dt,
                "open": _to_float(bar.get("o")),
                "high": _to_float(bar.get("h")),
                "low": _to_float(bar.get("l")),
                "close": close,
                "volume": _to_float(bar.get("v")),
                "amount": _to_float(bar.get("a")),
            })
    return rows


def _minute_frame(bars: dict) -> pl.DataFrame:
    rows = _minute_rows(bars)
    # 显式 schema: 稀疏列(某批全为 None)不会被推断成 Null 后 append 失败(#458 同修)
    return pl.DataFrame(rows, schema=_MINUTE_SCHEMA) if rows else pl.DataFrame()


def _daily_rows(payload: dict) -> list[dict]:
    """amber daily 矩阵 → 长表行 [symbol, date, open..close, volume, amount]。

    matrix[d][c] 按日期×代码对齐; 全 None 的 cell(停牌/未收录)跳过。
    结构整体变化(codes/dates/矩阵列缺失)打告警并按空数据处理, 不静默伪造。
    """
    if not isinstance(payload, dict) or not payload.get("codes"):
        return []
    codes = payload["codes"]
    dates = payload.get("dates") or []
    matrices = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        m = payload.get(col)
        if not isinstance(m, list) or len(m) != len(dates):
            logger.warning("amber daily 响应结构异常: %s 矩阵与 dates 长度不符, 按空数据处理", col)
            return []
        matrices[col] = m
    rows: list[dict] = []
    for di, day in enumerate(dates):
        for ci, code in enumerate(codes):
            cells = {col: matrices[col][di][ci] if ci < len(matrices[col][di]) else None
                     for col in matrices}
            if cells["open"] is None and cells["close"] is None:
                continue  # 停牌/未收录: 无行, 不伪造
            rows.append({
                "symbol": str(code),
                "date": str(day),
                "open": _to_float(cells["open"]),
                "high": _to_float(cells["high"]),
                "low": _to_float(cells["low"]),
                "close": _to_float(cells["close"]),
                "volume": _to_float(cells["volume"]) or 0.0,
                "amount": _to_float(cells["amount"]) or 0.0,
            })
    return rows


def _daily_batch_size(start_d, end_d) -> int:
    """按窗口跨度计算单请求 codes 数: cell 预算 30 万 ÷ 估算交易日数。

    交易日 ≈ 自然日 × 5/7; 1 年窗口 → ~1000 只/批, 10 年 → ~115 只/批,
    保证深窗口整段拉回而不触发服务端 400k cell 截断。
    """
    span_days = max((end_d - start_d).days, 1)
    trade_days_est = max(span_days * 5 // 7, 1)
    return max(60, min(_DAILY_CODES_MAX, _DAILY_CELL_BUDGET // trade_days_est))


_SNAPSHOT_CORE_COLS = ("pre", "last", "o", "h", "l", "vol", "amt")


def _snapshot_records(payload: dict, fetched_ms: int) -> list[dict]:
    """amber snapshot 列式矩阵 → realtime record 列表 (字段契约同 fuyao 映射)。

    核心列(pre/last/o/h/l/vol/amt)必须与代码列 c 等长; 任一缺失/错长视为
    结构整体变化 → 告警并返回 [](不静默、不伪造)。b1 等盘口列本插件不用。
    last 无效(≤0 或缺失)的行跳过: 该行情未成交, 保留会推导出 -100% 的伪涨跌。
    """
    codes = payload.get("c") if isinstance(payload, dict) else None
    if not isinstance(codes, list) or not codes:
        return []
    columns: dict[str, list] = {}
    for col in _SNAPSHOT_CORE_COLS:
        arr = payload.get(col)
        if not isinstance(arr, list) or len(arr) != len(codes):
            got = f"len={len(arr)}" if isinstance(arr, list) else type(arr).__name__
            logger.warning(
                "amber snapshot 响应结构异常: %s 列缺失或与代码列不等长 (got %s, want %d), 按空数据处理",
                col, got, len(codes),
            )
            return []
        columns[col] = arr
    if payload.get("stale"):
        logger.info("amber snapshot 标记 stale(采集滞后), 照常下发由下游新鲜度门控")
    records: list[dict] = []
    for i, code in enumerate(codes):
        symbol = str(code).strip()
        last = _to_float(columns["last"][i])
        prev = _to_float(columns["pre"][i])
        if not symbol or last is None or last <= 0:
            continue
        change_amount = last - prev if prev is not None else None
        change_pct = (change_amount / prev) if (change_amount is not None and prev > 0) else None
        records.append({
            "symbol": symbol,
            "name": None,  # 快照无名称, 下游用标的维表关联
            "last_price": last,
            "prev_close": prev,
            "open": _to_float(columns["o"][i]),
            "high": _to_float(columns["h"][i]),
            "low": _to_float(columns["l"][i]),
            "volume": _to_float(columns["vol"][i]),   # 手: amber 口径与契约一致, 直接透传
            "amount": _to_float(columns["amt"][i]),   # 元
            "change_pct": change_pct,                 # 小数制 (推导即小数, 无 /100)
            "change_amount": change_amount,
            "amplitude": None,      # 快照未提供, 不启发式计算
            "turnover_rate": None,  # 需股本口径, 交给 enriched 管道
            "timestamp": fetched_ms,  # 快照无逐行时间戳, 本地时刻兜底(契约允许)
            "session": None,
        })
    return records


class AmberProvider:
    name = "amber"
    builtin = True

    def __init__(self) -> None:
        self.config = _AmberConfig()
        self._client: amber_client.AmberClient | None = None

    def close(self) -> None:  # loader.load_all 重建注册表时会对每个 provider 调 close
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    def _get_client(self) -> amber_client.AmberClient:
        if self._client is None:
            self._client = amber_client.AmberClient(api_key=get_api_key(), base_url=get_base_url())
        return self._client

    # ---- daily ----
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """A 股日K → 内部契约。原始价、volume 手、amount 元, 均直接透传。"""
        chunks = [
            df
            for df in self.iter_daily(
                symbols,
                start_time=start_time,
                end_time=end_time,
                asset_type=asset_type,
                on_chunk_done=on_chunk_done,
            )
            if not df.is_empty()
        ]
        return pl.concat(chunks, how="diagonal_relaxed") if chunks else pl.DataFrame()

    def iter_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """分批产出日K, 供历史同步逐批落盘。批次按窗口跨度动态计算(cell 预算)。"""
        if not symbols:
            return
        if asset_type == "index":
            # amber 无指数数据(标的清单仅 stock/etf); 指数日K由 index_sync 走 TickFlow
            logger.info("amber 不提供指数日K, 跳过 %d 个标的", len(symbols))
            return
        end_dt = end_time or datetime.now()
        start_dt = start_time or (end_dt - timedelta(days=365))
        start_d, end_d = start_dt.date(), end_dt.date()
        if start_d > end_d:
            return

        client = self._get_client()
        batch_size = _daily_batch_size(start_d, end_d)
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        total = len(batches)
        for i, batch in enumerate(batches):
            rows: list[dict] = []
            try:
                payload = client.daily(batch, start=start_d.isoformat(), end=end_d.isoformat())
                if payload.get("truncated"):
                    logger.warning(
                        "amber daily 第 %d/%d 批被服务端截断(窗口 cell 超限), 最早日期数据缺失",
                        i + 1, total,
                    )
                rows = _daily_rows(payload)
            except AmberError as e:
                logger.warning("amber daily 第 %d/%d 批失败: %s", i + 1, total, e)
            df = normalize_daily(rows, source=self.name) if rows else pl.DataFrame()
            if on_chunk_done:
                on_chunk_done(i + 1, total)  # 空批次也推进, 保证最终 cur == total
            if not df.is_empty():
                yield df
            if i + 1 < total:
                time.sleep(_INTERVAL_S)

    # ---- full_minute ----
    def get_intraday_batch(
        self, symbols: list[str], count: int = 300, asset_type: str = "stock",
    ) -> pl.DataFrame:
        """全天修复轮: 给定标的当日 1 分钟K (canonical 8 列)。

        以 start=当日 00:00 窗口拉取(而非 count 回溯), 确保不含历史分钟 —
        _write_minute_partition 按行日期分区, 历史 bar 混入会放大写放大。
        """
        if not symbols:
            return pl.DataFrame()
        client = self._get_client()
        start_iso = f"{cn_today().isoformat()}T00:00:00"
        frames: list[pl.DataFrame] = []
        for i in range(0, len(symbols), _MINUTE_CODES_PER_REQ):
            batch = symbols[i:i + _MINUTE_CODES_PER_REQ]
            try:
                payload = client.minutes(
                    codes=batch, start=start_iso, count=MINUTES_COUNT_MAX,
                )
            except AmberError as e:
                logger.warning("amber 分钟批量 %d 只失败: %s", len(batch), e)
                continue
            if payload.get("truncated"):
                logger.warning("amber 分钟批量被服务端截断(bar 总数超 200k): %d 只", len(batch))
            df = _minute_frame(payload.get("bars") or {})
            if not df.is_empty():
                frames.append(df.filter(pl.col("datetime").dt.date() == cn_today()))
        return pl.concat(frames) if frames else pl.DataFrame()

    def get_intraday_latest(self, symbols: list[str] | None = None, count: int = 3) -> pl.DataFrame:
        """稳态增量轮: 每只标的最新 count 根分钟K。

        symbols=None 时按市场分段全市场拉取(SH/SZ/BJ 股票, 各 1 请求);
        每段 bar 量 = 标的数 × count ≪ 200k 上限。ETF 分段刻意不拉:
        增量轮落盘目标是股票表 kline_minute, ETF 分钟由盘后同步独立覆盖。
        """
        client = self._get_client()
        frames: list[pl.DataFrame] = []
        if symbols:
            for i in range(0, len(symbols), MINUTES_CODES_MAX):
                batch = symbols[i:i + MINUTES_CODES_MAX]
                try:
                    payload = client.minutes(codes=batch, count=count)
                except AmberError as e:
                    logger.warning("amber 分钟增量 %d 只失败: %s", len(batch), e)
                    continue
                df = _minute_frame(payload.get("bars") or {})
                if not df.is_empty():
                    frames.append(df)
        else:
            for market in ("SH", "SZ", "BJ"):
                try:
                    payload = client.minutes(market=market, type_="stock", count=count)
                except AmberError as e:
                    # BJ 段在部分部署无北交所数据(503/空) → 记日志继续, 不拖垮整轮
                    logger.warning("amber 分钟增量 %s 段失败: %s", market, e)
                    continue
                df = _minute_frame(payload.get("bars") or {})
                if not df.is_empty():
                    frames.append(df)
        return pl.concat(frames) if frames else pl.DataFrame()

    # ---- realtime ----
    def get_realtime(self) -> list[dict]:
        """全市场实时快照 → realtime records (quote_service 全市场模式轮询调用)。

        软失败: 任何异常返回 [] + warning, 不阻断轮询线程 (契约语义)。
        指数补充(get_realtime_indices)刻意未实现: amber 无指数行情,
        指数由 quote_service 本地日K兜底接管。
        """
        try:
            payload = self._get_client().snapshot()
        except AmberError as e:
            logger.warning("amber 实时快照拉取失败: %s", e)
            return []
        records = _snapshot_records(payload, int(time.time() * 1000))
        if not records and payload.get("n"):
            # 服务端声称有数据但一行都没映射出来 → 结构漂移, 已在映射层告警
            logger.warning("amber 快照映射 0 行 (服务端 n=%s)", payload.get("n"))
        return records

    # ---- 设置页试拉 ----
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        out = {"provider": self.name, "dataset": dataset}
        if dataset not in _DATASETS:
            out["error"] = f"amber 未提供 {dataset} 数据集, 该能力回退 TickFlow"
            return out
        try:
            client = self._get_client()
            if dataset == "daily":
                probe_sym = (symbols or ["600519.SH"])[0]
                end_d = cn_today()
                payload = client.daily([probe_sym], start=(end_d - timedelta(days=10)).isoformat())
                df = normalize_daily(_daily_rows(payload), source=self.name)
            elif dataset == "realtime":
                records = _snapshot_records(client.snapshot(), int(time.time() * 1000))
                out.update(rows=len(records), columns=["symbol", "last_price", "prev_close"],
                           preview=records[:3])
                return out
            else:
                probe_sym = (symbols or ["600519.SH"])[0]
                payload = client.minutes(
                    codes=[probe_sym], count=30,
                    start=f"{cn_today().isoformat()}T00:00:00",
                )
                df = _minute_frame(payload.get("bars") or {})
            out.update(
                rows=df.height,
                columns=list(df.columns),
                preview=df.head(3).to_dicts(),
            )
        except AmberError as e:
            out["error"] = str(e)
        return out
