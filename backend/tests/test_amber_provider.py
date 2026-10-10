"""amber 插件契约测试 (不依赖真实网络与 API Key)。

覆盖 docs/plugin-development.md 的插件 PR 测试要求: 字段映射与单位、响应结构
变体告警、分块与进度回调、软失败隔离、能力声明、Key 语义、loader 集成。
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from app.plugins.amber import client as ac
from app.plugins.amber import provider as ap
from app.plugins.amber.client import AmberError

# ---------------------------------------------------------------- helpers


class _FakeAmber:
    """按调用序回放预设响应的假客户端, 记录全部请求供断言。"""

    def __init__(self, *, daily=None, minutes=None, instruments=None,
                 snapshot=None, snapshot_error=None,
                 daily_error=None, minutes_error=None):
        self._daily = list(daily or [])
        self._minutes = list(minutes or [])
        self._instruments = instruments or {"n": 2319, "items": []}
        self._snapshot = snapshot
        self._snapshot_error = snapshot_error
        self._daily_error = list(daily_error or [])
        self._minutes_error = list(minutes_error or [])
        self.daily_calls: list[dict] = []
        self.minutes_calls: list[dict] = []
        self.instruments_calls: list[dict] = []
        self.snapshot_calls: list[dict] = []

    def daily(self, codes, start=None, end=None):
        self.daily_calls.append({"codes": list(codes), "start": start, "end": end})
        if self._daily_error:
            err = self._daily_error.pop(0)
            if err is not None:
                raise err
        return self._daily.pop(0) if self._daily else {"codes": [], "dates": []}

    def minutes(self, codes=None, *, start=None, end=None, count=240,
                market=None, type_=None):
        self.minutes_calls.append({
            "codes": list(codes) if codes else None, "start": start, "end": end,
            "count": count, "market": market, "type": type_,
        })
        if self._minutes_error:
            err = self._minutes_error.pop(0)
            if err is not None:
                raise err
        return self._minutes.pop(0) if self._minutes else {"n": 0, "total": 0, "bars": {}}

    def snapshot(self, market=None, type_=None):
        self.snapshot_calls.append({"market": market, "type": type_})
        if self._snapshot_error is not None:
            raise self._snapshot_error
        return self._snapshot or {"n": 0, "c": []}

    def instruments(self, market=None, type_=None):
        self.instruments_calls.append({"market": market, "type": type_})
        return self._instruments

    def close(self):
        pass


def _provider_with(monkeypatch, fake: _FakeAmber) -> ap.AmberProvider:
    monkeypatch.setattr(ap, "amber_client", type("M", (), {"AmberClient": lambda **kw: fake}))
    monkeypatch.setattr(ap, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ap, "get_base_url", lambda: "http://amber.test")
    return ap.AmberProvider()


def _bar(ts="2026-10-08T09:35:00", o=10.0, h=10.5, lo=9.8, c=10.2, v=123, a=1_234_000.0):
    return {"ts": ts, "o": o, "h": h, "l": lo, "c": c, "v": v, "a": a}


def _daily_payload(codes, dates, cells):
    """cells: {(date, code): (open, high, low, close, volume, amount)}, 缺失即停牌 None。"""
    def matrix(idx):
        return [[cells.get((d, c), (None,) * 6)[idx] for c in codes] for d in dates]
    return {
        "codes": codes, "dates": dates, "n": len(dates),
        "open": matrix(0), "high": matrix(1), "low": matrix(2),
        "close": matrix(3), "volume": matrix(4), "amount": matrix(5),
        "quality": [[None] * len(codes) for _ in dates],
        "truncated": False,
    }


def _snapshot_payload(codes, rows):
    """列式快照: rows = {code: dict(pre=.., last=.., o=.., h=.., l=.., vol=.., amt=..)}。"""
    def col(k, default=None):
        return [rows.get(c, {}).get(k, default) for c in codes]
    return {
        "v": 7, "day": "2026-10-08", "stale": False, "n": len(codes), "c": list(codes),
        "pre": col("pre"), "last": col("last"), "o": col("o"), "h": col("h"),
        "l": col("l"), "vol": col("vol"), "amt": col("amt"),
        "b1": [0.0] * len(codes), "bv1": [0] * len(codes),
        "a1": [0.0] * len(codes), "av1": [0] * len(codes),
        "in": [0] * len(codes), "out": [0] * len(codes),
    }


_TODAY = date(2026, 10, 8)


@pytest.fixture
def today(monkeypatch):
    monkeypatch.setattr(ap, "cn_today", lambda: _TODAY)
    return _TODAY


# ---------------------------------------------------------------- client


def _patch_http(monkeypatch, payload, status_code=200, *, is_json=True):
    class _Resp:
        def json(self):
            if not is_json:
                raise ValueError("no json")
            return payload

    _Resp.status_code = status_code

    captured = {}

    class _Http:
        def request(self, method, path, json=None, params=None):
            captured.update({"method": method, "path": path, "json": json, "params": params})
            return _Resp()

        def close(self):
            pass

    monkeypatch.setattr(ac.httpx, "Client", lambda **kw: _Http())
    return captured


def test_client_http_error_envelope(monkeypatch):
    cap = _patch_http(monkeypatch, {"detail": "missing bearer key"}, status_code=401)
    client = ac.AmberClient(api_key="k")
    with pytest.raises(AmberError, match="missing bearer key"):
        client.daily(["600519.SH"])
    assert cap["path"] == "/v1/daily"


def test_client_rejects_non_dict_payload(monkeypatch):
    _patch_http(monkeypatch, [1, 2, 3])
    client = ac.AmberClient(api_key="k")
    with pytest.raises(AmberError, match="非对象"):
        client.daily(["600519.SH"])


def test_client_minutes_clamps_count_and_codes(monkeypatch):
    cap = _patch_http(monkeypatch, {"bars": {}})
    client = ac.AmberClient(api_key="k")
    client.minutes(codes=[f"{i:06d}.SH" for i in range(2500)], count=99_999)
    assert cap["json"]["count"] == ac.MINUTES_COUNT_MAX
    assert len(cap["json"]["codes"]) == ac.MINUTES_CODES_MAX


def test_client_instruments_uses_query_params(monkeypatch):
    cap = _patch_http(monkeypatch, {"n": 0, "items": []})
    client = ac.AmberClient(api_key="k")
    client.instruments(market="SH", type_="stock")
    assert cap["method"] == "GET"
    assert cap["params"] == {"market": "SH", "type": "stock"}
    assert cap["json"] is None


def test_client_snapshot_full_market_no_filters(monkeypatch):
    cap = _patch_http(monkeypatch, {"n": 0, "c": []})
    client = ac.AmberClient(api_key="k")
    client.snapshot()
    assert cap["method"] == "GET"
    assert cap["path"] == "/v1/snapshot"
    assert cap["params"] is None and cap["json"] is None


# ---------------------------------------------------------------- daily


def test_daily_matrix_mapping_and_suspension_skip(monkeypatch):
    """矩阵 → 长表; 全 None cell(停牌)跳过; 手/元零换算直接透传。"""
    payload = _daily_payload(
        ["600519.SH", "000001.SZ"],
        ["2026-09-30", "2026-10-08"],
        {
            ("2026-09-30", "600519.SH"): (1700.0, 1710.0, 1690.0, 1705.0, 26000, 4_433_000_000.0),
            ("2026-10-08", "600519.SH"): (1705.0, 1720.0, 1700.0, 1718.0, 31000, 5_325_800_000.0),
            ("2026-10-08", "000001.SZ"): (11.0, 11.2, 10.9, 11.1, 800_000, 88_800_000.0),
        },
    )
    provider = _provider_with(monkeypatch, _FakeAmber(daily=[payload]))
    df = provider.get_daily(
        ["600519.SH", "000001.SZ"],
        start_time=datetime(2026, 9, 1), end_time=datetime(2026, 10, 8),
    )
    assert df.height == 3  # 2026-09-30 平安银行停牌 cell 被跳过
    row = df.filter(
        (pl.col("symbol") == "600519.SH") & (pl.col("date") == date(2026, 10, 8))
    ).to_dicts()[0]
    assert row["open"] == 1705.0 and row["close"] == 1718.0
    # volume 手 / amount 元: 原值透传 (不 /100 不 *100)
    assert row["volume"] == 31000.0
    assert row["amount"] == 5_325_800_000.0


def test_daily_truncated_warns_but_yields(monkeypatch, caplog):
    payload = _daily_payload(
        ["600519.SH"], ["2026-10-08"],
        {("2026-10-08", "600519.SH"): (1.0, 1.0, 1.0, 1.0, 1, 1.0)},
    )
    payload["truncated"] = True
    provider = _provider_with(monkeypatch, _FakeAmber(daily=[payload]))
    with caplog.at_level("WARNING"):
        df = provider.get_daily(["600519.SH"], None, None)
    assert df.height == 1
    assert any("截断" in r.message for r in caplog.records)


def test_iter_daily_batches_and_chunk_progress(monkeypatch):
    # 桩掉批大小计算(真实函数有 60 只下限), 强制 3 标的 → 2 批
    monkeypatch.setattr(ap, "_daily_batch_size", lambda s, e: 2)
    payload = _daily_payload(
        ["600519.SH"], ["2026-10-08"],
        {("2026-10-08", "600519.SH"): (1.0, 1.0, 1.0, 1.0, 1, 1.0)},
    )
    fake = _FakeAmber(daily=[payload, {"codes": [], "dates": []}])
    provider = _provider_with(monkeypatch, fake)
    progress: list[tuple[int, int]] = []
    frames = list(provider.iter_daily(
        ["600519.SH", "000001.SZ", "300750.SZ"],
        start_time=datetime(2026, 9, 1), end_time=datetime(2026, 10, 8),
        on_chunk_done=lambda cur, total: progress.append((cur, total)),
    ))
    assert len(fake.daily_calls) == 2
    # 空批次也推进进度 → 最终 cur == total
    assert progress == [(1, 2), (2, 2)]
    assert sum(f.height for f in frames) == 1
    # 请求窗口按 ISO 日期透传
    assert fake.daily_calls[0]["start"] == "2026-09-01"
    assert fake.daily_calls[0]["end"] == "2026-10-08"


def test_iter_daily_batch_size_scales_with_window():
    near = ap._daily_batch_size(date(2026, 9, 28), date(2026, 10, 8))   # ~7 个交易日
    deep = ap._daily_batch_size(date(2016, 10, 8), date(2026, 10, 8))   # ~10 年
    assert near == ap._DAILY_CODES_MAX
    assert 60 <= deep < 200  # 深窗口显著收窄, 远离 400k cell 上限


def test_daily_partial_batch_failure_isolated(monkeypatch):
    monkeypatch.setattr(ap, "_daily_batch_size", lambda s, e: 2)
    payload = _daily_payload(
        ["300750.SZ"], ["2026-10-08"],
        {("2026-10-08", "300750.SZ"): (200.0, 200.0, 200.0, 200.0, 100, 20_000.0)},
    )
    fake = _FakeAmber(daily=[payload], daily_error=[AmberError("HTTP 503: warehouse 未启用")])
    provider = _provider_with(monkeypatch, fake)
    frames = list(provider.iter_daily(
        ["600519.SH", "000001.SZ", "300750.SZ"], None, None,
    ))
    assert [f.height for f in frames] == [1]  # 第 1 批失败, 第 2 批照常产出


def test_daily_empty_payload_returns_empty(monkeypatch):
    provider = _provider_with(monkeypatch, _FakeAmber(daily=[{"codes": [], "dates": []}]))
    df = provider.get_daily(["600519.SH"], None, None)
    assert df.is_empty()


def test_daily_structure_change_warns_not_silent(monkeypatch, caplog):
    bad = {"codes": ["600519.SH"], "dates": ["2026-10-08"], "open": [[1.0]]}  # 缺 high/low/...
    provider = _provider_with(monkeypatch, _FakeAmber(daily=[bad]))
    with caplog.at_level("WARNING"):
        df = provider.get_daily(["600519.SH"], None, None)
    assert df.is_empty()
    assert any("结构异常" in r.message for r in caplog.records)


def test_daily_index_asset_type_skipped(monkeypatch):
    fake = _provider_with(monkeypatch, _FakeAmber())
    frames = list(fake.iter_daily(["000001.SH"], None, None, asset_type="index"))
    assert frames == []


# ---------------------------------------------------------------- full_minute


def test_intraday_batch_maps_naive_wallclock(monkeypatch, today):
    fake = _FakeAmber(minutes=[{"n": 1, "total": 1, "bars": {"600519.SH": [
        _bar(), _bar(ts="2026-10-08T14:55:00", c=11.0),
    ]}}])
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_batch(["600519.SH"])
    assert df.height == 2
    row = df.to_dicts()[0]
    assert row["datetime"] == datetime(2026, 10, 8, 9, 35)
    assert row["volume"] == 123.0 and row["amount"] == 1_234_000.0  # 手/元透传
    assert list(df.columns) == list(ap._MINUTE_SCHEMA)


def test_intraday_batch_normalizes_tzaware(monkeypatch, today):
    fake = _FakeAmber(minutes=[{"bars": {"600519.SH": [_bar(ts="2026-10-08T09:35:00+08:00")]}}])
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_batch(["600519.SH"])
    assert df.to_dicts()[0]["datetime"] == datetime(2026, 10, 8, 9, 35)


def test_intraday_batch_requests_today_window_and_chunks(monkeypatch, today):
    """修复轮: start=当日 00:00 + count=24000; 超过 700 只自动分块。"""
    syms = [f"{600000 + i}.SH" for i in range(1500)]
    fake = _FakeAmber(minutes=[{"bars": {}}] * 3)
    provider = _provider_with(monkeypatch, fake)
    provider.get_intraday_batch(syms)
    assert len(fake.minutes_calls) == 3
    assert [len(c["codes"]) for c in fake.minutes_calls] == [700, 700, 100]
    for call in fake.minutes_calls:
        assert call["start"] == "2026-10-08T00:00:00"
        assert call["count"] == ac.MINUTES_COUNT_MAX


def test_intraday_batch_filters_history_bars(monkeypatch, today):
    """_write_minute_partition 按行日期分区 — 历史 bar 必须在 provider 侧滤掉。"""
    fake = _FakeAmber(minutes=[{"bars": {"600519.SH": [
        _bar(ts="2026-10-07T14:55:00", c=99.0),   # 昨日尾根 (count 窗口回溯混入)
        _bar(ts="2026-10-08T09:31:00", c=10.0),
        _bar(ts="2026-10-08T09:32:00", c=10.1),
    ]}}])
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_batch(["600519.SH"])
    assert df.height == 2
    assert df["datetime"].min() == datetime(2026, 10, 8, 9, 31)


def test_intraday_batch_skips_bad_rows(monkeypatch, today):
    fake = _FakeAmber(minutes=[{"bars": {"600519.SH": [
        _bar(c=None),                       # close 缺失
        _bar(ts="not-a-date"),              # ts 无法解析
        {"ts": "2026-10-08T09:35:00"},      # 缺全部数值字段
        _bar(c=10.5),                       # 正常
    ]}}])
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_batch(["600519.SH"])
    assert df.height == 1
    assert df.to_dicts()[0]["close"] == 10.5


def test_intraday_batch_chunk_failure_isolated(monkeypatch, today):
    fake = _FakeAmber(
        minutes=[{"bars": {"600519.SH": [_bar()]}}],
        minutes_error=[AmberError("HTTP 503")],
    )
    provider = _provider_with(monkeypatch, fake)
    syms = ["600519.SH", "600001.SH"] + [f"{600100 + i}.SH" for i in range(700)]
    df = provider.get_intraday_batch(syms)  # 第 1 批(700 只)失败, 第 2 批产出
    assert df.height == 1


def test_intraday_latest_whole_market_segments(monkeypatch, today):
    """symbols=None → SH/SZ/BJ 股票分段 (刻意不含 etf, 增量轮只写股票表)。"""
    fake = _FakeAmber(minutes=[
        {"bars": {"600519.SH": [_bar()]}},
        {"bars": {"000001.SZ": [_bar()]}},
        {"bars": {}},
    ])
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_latest(count=3)
    assert df.height == 2
    assert [c["market"] for c in fake.minutes_calls] == ["SH", "SZ", "BJ"]
    assert all(c["type"] == "stock" for c in fake.minutes_calls)
    assert all(c["count"] == 3 for c in fake.minutes_calls)


def test_intraday_latest_with_symbols_chunks_2000(monkeypatch, today):
    fake = _FakeAmber(minutes=[{"bars": {}}] * 2)
    provider = _provider_with(monkeypatch, fake)
    provider.get_intraday_latest(symbols=[f"{i:06d}.SH" for i in range(2500)], count=5)
    assert [len(c["codes"]) for c in fake.minutes_calls] == [2000, 500]
    assert all(c["count"] == 5 for c in fake.minutes_calls)


def test_intraday_latest_segment_failure_isolated(monkeypatch, today):
    fake = _FakeAmber(
        minutes=[{"bars": {"600519.SH": [_bar()]}}, {"bars": {"000001.SZ": [_bar(c=11.0)]}}],
        minutes_error=[None, None, AmberError("HTTP 503")],  # BJ 段失败
    )
    provider = _provider_with(monkeypatch, fake)
    df = provider.get_intraday_latest(count=3)
    assert df.height == 2


# ---------------------------------------------------------------- realtime


def test_get_realtime_columnar_mapping_and_units(monkeypatch):
    """列式快照 → realtime record; 手/元零换算; change_pct 小数制推导。"""
    payload = _snapshot_payload(
        ["600519.SH", "510300.SH"],
        {
            "600519.SH": {"pre": 1700.0, "last": 1717.0, "o": 1705.0, "h": 1720.0,
                          "l": 1700.0, "vol": 31000, "amt": 5_325_800_000.0},
            "510300.SH": {"pre": 4.10, "last": 4.08, "o": 4.11, "h": 4.12,
                          "l": 4.07, "vol": 900_000, "amt": 367_200_000.0},
        },
    )
    provider = _provider_with(monkeypatch, _FakeAmber(snapshot=payload))
    records = provider.get_realtime()
    assert [r["symbol"] for r in records] == ["600519.SH", "510300.SH"]  # ETF 也在池内
    row = records[0]
    assert row["last_price"] == 1717.0 and row["prev_close"] == 1700.0
    assert row["volume"] == 31000.0 and row["amount"] == 5_325_800_000.0  # 手/元透传
    assert row["change_amount"] == 17.0
    assert row["change_pct"] == pytest.approx(17.0 / 1700.0)  # 小数制, 非 1.0
    assert row["name"] is None and row["timestamp"] > 0
    assert isinstance(row["timestamp"], int)


def test_get_realtime_skips_invalid_last_rows(monkeypatch):
    """last ≤0 / 缺失的行(未成交)跳过 — 保留会推导出 -100% 伪涨跌。"""
    payload = _snapshot_payload(
        ["600519.SH", "300750.SZ", "000001.SZ"],
        {
            "600519.SH": {"pre": 1700.0, "last": 1717.0, "vol": 1, "amt": 1.0},
            "300750.SZ": {"pre": 200.0, "last": 0, "vol": 0, "amt": 0.0},      # 零价
            "000001.SZ": {"pre": 11.0, "last": None, "vol": 0, "amt": 0.0},    # 缺价
        },
    )
    provider = _provider_with(monkeypatch, _FakeAmber(snapshot=payload))
    records = provider.get_realtime()
    assert [r["symbol"] for r in records] == ["600519.SH"]


def test_get_realtime_structure_change_warns(monkeypatch, caplog):
    bad = {"v": 7, "n": 2, "c": ["600519.SH", "000001.SZ"], "pre": [1700.0, 11.0]}
    # last/o/h/l/vol/amt 全缺失 → 结构异常
    provider = _provider_with(monkeypatch, _FakeAmber(snapshot=bad))
    with caplog.at_level("WARNING"):
        records = provider.get_realtime()
    assert records == []
    assert any("结构异常" in r.message for r in caplog.records)


def test_get_realtime_soft_failure_returns_empty(monkeypatch, caplog):
    provider = _provider_with(monkeypatch, _FakeAmber(
        snapshot_error=AmberError("HTTP 503: 数据文件不可用")))
    with caplog.at_level("WARNING"):
        records = provider.get_realtime()
    assert records == []  # 软失败: 不抛异常, 不阻断轮询线程
    assert any("实时快照拉取失败" in r.message for r in caplog.records)


def test_get_realtime_stale_flag_still_delivers(monkeypatch):
    """stale(采集滞后)照常下发 — 新鲜度门控是下游职责(与 TickFlow 收盘后同语义)。"""
    payload = _snapshot_payload(
        ["600519.SH"], {"600519.SH": {"pre": 1700.0, "last": 1717.0, "vol": 1, "amt": 1.0}},
    )
    payload["stale"] = True
    provider = _provider_with(monkeypatch, _FakeAmber(snapshot=payload))
    assert len(provider.get_realtime()) == 1


def test_get_realtime_no_indices_method():
    """amber 无指数行情 → 刻意不实现 get_realtime_indices, 指数走本地日K兜底。"""
    provider = ap.AmberProvider()
    assert not callable(getattr(provider, "get_realtime_indices", None))


# ---------------------------------------------------------------- key & capability


def test_availability_two_states(monkeypatch):
    monkeypatch.setattr(ap, "get_api_key", lambda: "")
    ok, reason = ap.availability()
    assert not ok and "AMBER_API_KEY" in reason
    monkeypatch.setattr(ap, "get_api_key", lambda: "k")
    assert ap.availability() == (True, "ok")


def test_probe_api_key(monkeypatch):
    fake = _FakeAmber()
    monkeypatch.setattr(ap, "amber_client", type("M", (), {"AmberClient": lambda **kw: fake}))
    monkeypatch.setattr(ap, "get_base_url", lambda: "http://amber.test")
    ok, reason = ap.probe_api_key("good")
    assert ok and "2319" in reason  # 探活成功并回报标的数

    boom = _FakeAmber()
    boom.instruments = lambda market=None, type_=None: (_ for _ in ()).throw(
        AmberError("HTTP 401: missing bearer key"))
    monkeypatch.setattr(ap, "amber_client", type("M", (), {"AmberClient": lambda **kw: boom}))
    ok, reason = ap.probe_api_key("bad")
    assert not ok and "amber.test" in reason


def test_datasets_declaration():
    provider = ap.AmberProvider()
    assert set(provider.config.datasets) == {"daily", "full_minute", "realtime"}
    assert "adj_factor" not in provider.config.datasets
    assert "minute" not in provider.config.datasets


def test_loader_registers_amber_when_key_present(monkeypatch):
    from app.data_providers.custom import loader

    providers_before = dict(loader._PROVIDERS)
    status_before = dict(loader._PLUGIN_STATUS)
    monkeypatch.setattr(ap, "get_api_key", lambda: "test-key")
    try:
        loader._load_builtin_plugins()
        assert loader._PLUGIN_STATUS["amber"]["available"] is True
        assert set(loader._PLUGIN_STATUS["amber"]["datasets"]) == {"daily", "full_minute", "realtime"}
        assert loader.provider_has_dataset("amber", "daily")
        assert loader.provider_has_dataset("amber", "full_minute")
        assert loader.provider_has_dataset("amber", "realtime")
        assert not loader.provider_has_dataset("amber", "minute")
        assert loader.get_provider("amber").name == "amber"
    finally:
        loader._PROVIDERS.clear()
        loader._PROVIDERS.update(providers_before)
        loader._PLUGIN_STATUS.clear()
        loader._PLUGIN_STATUS.update(status_before)


def test_test_dataset_unsupported(monkeypatch):
    provider = _provider_with(monkeypatch, _FakeAmber())
    out = provider.test_dataset("adj_factor")
    assert "回退" in out["error"]


def test_test_dataset_realtime_probe(monkeypatch):
    payload = _snapshot_payload(
        ["600519.SH"], {"600519.SH": {"pre": 1700.0, "last": 1717.0, "vol": 1, "amt": 1.0}},
    )
    provider = _provider_with(monkeypatch, _FakeAmber(snapshot=payload))
    out = provider.test_dataset("realtime")
    assert out["rows"] == 1 and "error" not in out
    assert out["preview"][0]["symbol"] == "600519.SH"


def test_test_dataset_daily_probe(monkeypatch, today):
    payload = _daily_payload(
        ["600519.SH"], ["2026-10-08"],
        {("2026-10-08", "600519.SH"): (1.0, 1.0, 1.0, 1.0, 1, 1.0)},
    )
    provider = _provider_with(monkeypatch, _FakeAmber(daily=[payload]))
    out = provider.test_dataset("daily")
    assert out["rows"] == 1 and "error" not in out
