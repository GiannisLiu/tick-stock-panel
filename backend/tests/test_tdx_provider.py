"""TdxProvider 契约测试(不依赖真实网络与通达信客户端)。

覆盖 CONTRIBUTING §9: 字段映射与单位口径、通达信返回结构变体(字符串/数字/短数组/
缺字段)、软失败(单标的失败不拖垮整批、无盘口不产出条目)、能力声明(未声明数据集回退
TickFlow)、availability 两态、清单注册。真实通达信交互只在 client 层, 用 _post 替换点隔离。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from app.plugins.tdx import client as tc
from app.plugins.tdx import provider as tp
from app.plugins.tdx.provider import TdxProvider


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    """把"当日"固定在 2026-09-21(周一), 让日期过滤断言与真实时钟无关。"""
    monkeypatch.setattr(tp, "cn_today", lambda: date(2026, 9, 21))
    return date(2026, 9, 21)


class _FakeClient:
    """按 symbol 返回预置快照, 记录调用; error 指定时所有调用抛 TdxError。"""

    def __init__(self, rows: dict[str, dict], error: Exception | None = None):
        self.rows = rows
        self.error = error
        self.calls: list[str] = []
        self.closed = False
        self.kline_calls: list[tuple[list[str], int]] = []
        self.codes: list[str] = []

    def snapshot(self, stock_code: str) -> dict:
        self.calls.append(stock_code)
        if self.error:
            raise self.error
        return dict(self.rows.get(stock_code) or {})

    def kline_1m(self, symbols: list[str], count: int) -> dict:
        self.kline_calls.append((list(symbols), count))
        if self.error:
            raise self.error
        return {s: dict(self.rows[s]) for s in symbols if s in self.rows}

    def market_codes(self) -> list[str]:
        return list(self.codes)

    def close(self) -> None:
        self.closed = True


def _min(symbols_dates: list[str], times: list[str] | None = None, **over):
    """通达信 1 分钟K 的列表式负载(每字段一个数组)。"""
    n = len(symbols_dates)
    row = {
        "Date": symbols_dates,
        "Time": times or ["93100", "93200"][:n],
        "Open": ["10.00"] * n,
        "High": ["10.50"] * n,
        "Low": ["9.90"] * n,
        "Close": ["10.20"] * n,
        "Volume": ["74800.00"] * n,  # 股
        "Amount": ["8825.67"] * n,  # 万元
        "ErrorId": "0",
    }
    row.update(over)
    return row


def _provider_with(monkeypatch, rows, error=None):
    fake = _FakeClient(rows, error=error)
    monkeypatch.setattr(
        tp, "tdx_client", SimpleNamespace(TdxClient=lambda **kw: fake, TdxError=tc.TdxError)
    )
    return TdxProvider(), fake


def _snap(buyp, buyv, sellp, sellv, **extra):
    return {"Buyp": buyp, "Buyv": buyv, "Sellp": sellp, "Sellv": sellv, **extra}


FULL_ROW = _snap(
    ["1252.57", "1252.50", "1252.40", "1252.30", "1252.20"],
    ["1", "7", "12", "20", "33"],
    ["1252.86", "1253.00", "1253.10", "1253.20", "1253.30"],
    ["57", "18", "9", "4", "2"],
    Now="1252.57",
    LastClose="1257.12",
    Volume="25016",
)


# ---- 字段映射、五档对齐与单位 ----


def test_depth5_maps_five_levels_and_units(monkeypatch):
    """五档价量按一档→五档排列; 盘口量直接透传(快照口径本身是手)。"""
    provider, _ = _provider_with(monkeypatch, {"600519.SH": FULL_ROW})
    data = provider.get_depth_batch(["600519.SH"])
    entry = data["600519.SH"]
    assert entry["bid_prices"] == [1252.57, 1252.50, 1252.40, 1252.30, 1252.20]
    assert entry["bid_volumes"] == [1, 7, 12, 20, 33]
    assert entry["ask_prices"] == [1252.86, 1253.00, 1253.10, 1253.20, 1253.30]
    assert entry["ask_volumes"] == [57, 18, 9, 4, 2]
    assert isinstance(entry["timestamp"], int) and entry["timestamp"] > 1_700_000_000_000


def test_empty_levels_become_zero_not_none(monkeypatch):
    """通达信空档位返回 "0.00"/"0" → 0, 保住 sealed 的 ask1 == 0 语义。"""
    row = _snap(
        ["10.00", "0.00", "0.00", "0.00", "0.00"],
        ["5", "0", "0", "0", "0"],
        ["0.00", "0.00", "0.00", "0.00", "0.00"],
        ["0", "0", "0", "0", "0"],
    )
    provider, _ = _provider_with(monkeypatch, {"301234.SZ": row})
    entry = provider.get_depth_batch(["301234.SZ"])["301234.SZ"]
    assert entry["ask_volumes"][0] == 0  # 涨停封板: 卖一空 → 真封
    assert entry["ask_prices"] == [0.0] * 5
    assert len(entry["bid_volumes"]) == 5


def test_sealed_down_shape_keeps_bid1_zero(monkeypatch):
    """跌停: 买一为 0、卖档有价 → bid1 == 0 可判真封。"""
    row = _snap(
        ["0.00"] * 5,
        ["0"] * 5,
        ["9.10", "9.11", "0.00", "0.00", "0.00"],
        ["100", "50", "0", "0", "0"],
    )
    provider, _ = _provider_with(monkeypatch, {"600001.SH": row})
    entry = provider.get_depth_batch(["600001.SH"])["600001.SH"]
    assert entry["bid_volumes"][0] == 0
    assert entry["ask_prices"][0] == 9.10


def test_no_book_at_all_is_skipped(monkeypatch):
    """五档全空(停牌/未订阅)不产出条目, 避免被误判为涨停真封。"""
    row = _snap(["0.00"] * 5, ["0"] * 5, ["0.00"] * 5, ["0"] * 5)
    provider, _ = _provider_with(monkeypatch, {"600002.SH": row})
    assert provider.get_depth_batch(["600002.SH"]) == {}


def test_missing_keys_and_short_arrays_are_padded(monkeypatch):
    """结构变体: 只有首档 / 数组缺失 / 非 list → 一律补齐到五档且不倒序。"""
    provider, _ = _provider_with(
        monkeypatch,
        {
            "600003.SH": {"Buyp": ["7.10"], "Buyv": ["3"]},  # 卖档整块缺失
            "600004.SH": {"Buyp": 7.1, "Buyv": 3, "Sellp": None, "Sellv": ["1", "2"]},
        },
    )
    data = provider.get_depth_batch(["600003.SH", "600004.SH"])
    assert data["600003.SH"]["bid_prices"] == [7.10, 0.0, 0.0, 0.0, 0.0]
    assert data["600003.SH"]["ask_volumes"] == [0] * 5
    assert data["600004.SH"]["bid_volumes"] == [0] * 5  # 标量不是合法五档形状 → 不臆造
    assert data["600004.SH"]["ask_volumes"] == [1, 2, 0, 0, 0]


def test_garbage_values_fall_back_to_zero(monkeypatch):
    row = _snap(["abc", None, "1.5"], ["x", "2"], ["--"], ["", "3"])
    provider, _ = _provider_with(monkeypatch, {"600005.SH": row})
    entry = provider.get_depth_batch(["600005.SH"])["600005.SH"]
    assert entry["bid_prices"] == [0.0, 0.0, 1.5, 0.0, 0.0]
    assert entry["bid_volumes"] == [0, 2, 0, 0, 0]
    assert entry["ask_volumes"] == [0, 3, 0, 0, 0]


# ---- 软失败 ----


def test_single_symbol_failure_does_not_break_batch(monkeypatch):
    class _Flaky(_FakeClient):
        def snapshot(self, stock_code):
            if stock_code == "600006.SH":
                raise tc.TdxError("通达信服务不可用")
            return super().snapshot(stock_code)

    fake = _Flaky({"600007.SH": FULL_ROW})
    monkeypatch.setattr(
        tp, "tdx_client", SimpleNamespace(TdxClient=lambda **kw: fake, TdxError=tc.TdxError)
    )
    data = TdxProvider().get_depth_batch(["600006.SH", "600007.SH"])
    assert list(data) == ["600007.SH"]


def test_all_symbols_fail_returns_empty_dict(monkeypatch):
    provider, _ = _provider_with(monkeypatch, {}, error=tc.TdxError("连接被拒绝"))
    assert provider.get_depth_batch(["600519.SH", "000001.SZ"]) == {}


def test_empty_symbols_short_circuits(monkeypatch):
    provider, fake = _provider_with(monkeypatch, {})
    assert provider.get_depth_batch([]) == {}
    assert fake.calls == []


def test_close_releases_client(monkeypatch):
    provider, fake = _provider_with(monkeypatch, {"600519.SH": FULL_ROW})
    provider.get_depth_batch(["600519.SH"])
    provider.close()
    assert fake.closed is True
    assert provider._client is None


# ---- 能力声明 ----


def test_datasets_declaration():
    """声明 depth5 / full_minute; 其余数据集 provider_has_dataset 为 False → 回退 tickflow。"""
    datasets = TdxProvider().config.datasets
    assert "depth5" in datasets
    assert "full_minute" in datasets
    for other in ("realtime", "daily", "adj_factor", "minute", "financial"):
        assert other not in datasets


# ---- availability 两态 ----


def _availability_monkeypatch(monkeypatch, available: bool):
    """provider 只依赖 client.ping(); ping 自身语义在 client 层单测。"""

    class _Stub:
        def ping(self) -> bool:
            return available

    monkeypatch.setattr(
        tp,
        "tdx_client",
        SimpleNamespace(
            TdxClient=lambda **kw: _Stub(),
            TdxError=tc.TdxError,
            default_base_url=tc.default_base_url,
        ),
    )


def test_availability_ok_when_client_running(monkeypatch):
    _availability_monkeypatch(monkeypatch, available=True)
    assert tp.availability() == (True, "ok")


def test_availability_false_when_client_down(monkeypatch):
    """客户端未运行(或未登录无行情): ping 为 False → 插件灰显, 不注册。"""
    _availability_monkeypatch(monkeypatch, available=False)
    ok, reason = tp.availability()
    assert ok is False and "未运行" in reason


def test_availability_never_raises(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        tp,
        "tdx_client",
        SimpleNamespace(
            TdxClient=_boom, TdxError=tc.TdxError, default_base_url=tc.default_base_url
        ),
    )
    ok, reason = tp.availability()
    assert ok is False and "探测异常" in reason


def test_availability_false_on_client_without_quotes(monkeypatch):
    """未登录时客户端在跑但行情恒为 0.00: client.ping 为 False, 同样不算可用。"""
    _availability_monkeypatch(monkeypatch, available=False)
    assert tp.availability()[0] is False


# ---- client 层: JSON-RPC 解包与错误语义 ----


class _RecordingClient(tc.TdxClient):
    def __init__(self, body: bytes | Exception):
        super().__init__()
        self.body = body
        self.payloads: list[dict] = []

    def _post(self, payload: bytes) -> bytes:
        self.payloads.append(json.loads(payload.decode("utf-8")))
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def test_client_unwraps_result_value():
    client = _RecordingClient(b'{"id":1,"result":{"ErrorId":"0","Value":{"Now":"1.23"}}}')
    assert client.snapshot("000001.SZ") == {"Now": "1.23"}
    assert client.payloads[0]["method"] == "get_market_snapshot"
    assert client.payloads[0]["params"] == {"stock_code": "000001.SZ"}


def test_client_raises_on_error_id():
    client = _RecordingClient(b'{"id":1,"result":{"ErrorId":"7","Error":"run_id invalid"}}')
    with pytest.raises(tc.TdxError):
        client.snapshot("000001.SZ")


def test_client_raises_on_rpc_error_object():
    client = _RecordingClient(b'{"id":1,"error":{"message":"connection lost"}}')
    with pytest.raises(tc.TdxError):
        client.snapshot("000001.SZ")


def test_client_raises_on_connection_failure():
    client = _RecordingClient(OSError("connection refused"))
    with pytest.raises(tc.TdxError):
        client.snapshot("000001.SZ")


def test_client_raises_on_non_json_body():
    client = _RecordingClient(b"<html>502</html>")
    with pytest.raises(tc.TdxError):
        client.snapshot("000001.SZ")


def test_client_base_url_from_env(monkeypatch):
    """Docker 部署: 用 TDX_BASE_URL 指向宿主机(host.docker.internal)。"""
    monkeypatch.setenv(tc.BASE_URL_ENV, "http://host.docker.internal:17709/")
    assert tc.TdxClient().base_url == "http://host.docker.internal:17709/"
    assert tc.default_base_url() == "http://host.docker.internal:17709/"
    monkeypatch.delenv(tc.BASE_URL_ENV, raising=False)
    assert tc.TdxClient().base_url == tc.DEFAULT_BASE_URL
    assert tc.TdxClient(base_url="http://10.0.0.5:17709/").base_url == "http://10.0.0.5:17709/"


def test_availability_hint_mentions_docker_when_loopback(monkeypatch):
    """未连通且地址是 loopback 时, 提示里给出 Docker 的改法。"""
    monkeypatch.delenv(tc.BASE_URL_ENV, raising=False)
    _availability_monkeypatch(monkeypatch, available=False)
    ok, reason = tp.availability()
    assert ok is False and "host.docker.internal" in reason


def test_client_ping_false_without_quotes():
    row = {"LastClose": "0.00", "Now": "0.00"}  # 未登录时通达信返回字符串 0.00
    client = _RecordingClient(
        json.dumps({"id": 1, "result": {"ErrorId": "0", "Value": row}}).encode()
    )
    assert client.ping() is False


def test_client_ping_true_with_quotes():
    row = {"LastClose": "11.70", "Now": "11.73"}
    client = _RecordingClient(
        json.dumps({"id": 1, "result": {"ErrorId": "0", "Value": row}}).encode()
    )
    assert client.ping() is True


# ---- 设置页试拉 ----


def test_test_dataset_depth5_preview(monkeypatch):
    provider, _ = _provider_with(monkeypatch, {"600519.SH": FULL_ROW})
    out = provider.test_dataset("depth5", ["600519.SH"])
    assert out["rows"] == 1
    assert out["columns"][0] == "symbol"
    assert out["preview"][0]["symbol"] == "600519.SH"
    assert out["preview"][0]["ask_volumes"] == [57, 18, 9, 4, 2]


def test_test_dataset_surfaces_client_error(monkeypatch):
    provider, _ = _provider_with(monkeypatch, {}, error=tc.TdxError("通达信服务不可用"))
    out = provider.test_dataset("depth5", ["600519.SH"])
    assert out["rows"] == 0 and "通达信" in out["error"]


def test_test_dataset_unsupported_dataset_reports_fallback(monkeypatch):
    provider, _ = _provider_with(monkeypatch, {})
    out = provider.test_dataset("realtime")
    assert out["rows"] == 0 and "回退 TickFlow" in out["error"]


# ---- full_minute: 当日 1 分钟K ----

DAY = "20260921"


def _minute_provider(monkeypatch, rows, error=None):
    provider, fake = _provider_with(monkeypatch, rows, error=error)
    return provider, fake


def test_full_minute_datetime_and_units(monkeypatch):
    """Date+Time → 北京墙钟 naive; Volume 股→手(÷100); Amount 万元→元(x10000)。"""
    rows = {"000001.SZ": _min([DAY, DAY], ["93100", "150000"])}
    provider, _ = _minute_provider(monkeypatch, rows)
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df.columns == [
        "symbol",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert df.height == 2
    assert df["datetime"].to_list() == [
        datetime(2026, 9, 21, 9, 31),
        datetime(2026, 9, 21, 15, 0),
    ]
    assert df.schema["datetime"] == pl.Datetime("us")
    assert df["volume"].to_list() == [748.0, 748.0]
    assert df["amount"].to_list() == [88_256_700.0, 88_256_700.0]
    assert df["close"].to_list() == [10.2, 10.2]


def test_full_minute_drops_other_days(monkeypatch):
    """冷缓存时接口会回历史 bar; 只保留当日, 避免把旧分钟线当当日污染存储。"""
    rows = {"000001.SZ": _min(["20260918", DAY, DAY], ["145900", "93100", "93200"])}
    provider, _ = _minute_provider(monkeypatch, rows)
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df.height == 2
    assert df["datetime"].to_list() == [datetime(2026, 9, 21, 9, 31), datetime(2026, 9, 21, 9, 32)]


def test_full_minute_only_old_days_yields_empty(monkeypatch):
    """全是旧日期(冷启动首轮) → 空帧, 让 minute_refresh 走空轮自愈。"""
    rows = {"000001.SZ": _min(["20260915", "20260915"])}
    provider, _ = _minute_provider(monkeypatch, rows)
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df.is_empty() and df.columns[0] == "symbol"


def test_full_minute_missing_amount_is_null_not_faked(monkeypatch):
    """接口不给成交额时置 null, 不得伪造。"""
    row = _min([DAY])
    del row["Amount"]
    provider, _ = _minute_provider(monkeypatch, {"000001.SZ": row})
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df["amount"].to_list() == [None]


def test_full_minute_malformed_time_is_dropped(monkeypatch):
    """时间字段畸形(93:99) → 丢掉该根而不是猜一个时间。"""
    rows = {"000001.SZ": _min([DAY, DAY], ["9399", "93100"])}
    provider, _ = _minute_provider(monkeypatch, rows)
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df.height == 1 and df["datetime"].to_list() == [datetime(2026, 9, 21, 9, 31)]


def test_full_minute_chunks_by_100(monkeypatch):
    """get_market_data 单请求上限 100 只, 超出的必须分块而不是静默丢掉。"""
    syms = [f"{i:06d}.SZ" for i in range(250)]
    provider, fake = _minute_provider(monkeypatch, {s: _min([DAY]) for s in syms})
    df = provider.get_intraday_batch(syms, count=240)
    assert [len(c) for c, _ in fake.kline_calls] == [100, 100, 50]
    assert df["symbol"].n_unique() == 250


def test_full_minute_clamps_count_to_one_day(monkeypatch):
    """当日最多 240 根, 向接口多要只会拉回前一日尾巴 → 压到 241。"""
    provider, fake = _minute_provider(monkeypatch, {"000001.SZ": _min([DAY])})
    provider.get_intraday_batch(["000001.SZ"], count=300)
    assert fake.kline_calls[0][1] == 241
    provider.get_intraday_latest(["000001.SZ"], count=3)
    assert fake.kline_calls[1][1] == 3


def test_get_intraday_latest_defaults_to_full_market(monkeypatch):
    """symbols=None → 取全市场代码当日缓存(增量轮语义), 维表只拉一次。"""
    provider, fake = _minute_provider(
        monkeypatch, {"000001.SZ": _min([DAY]), "600000.SH": _min([DAY])}
    )
    fake.codes = ["000001.SZ", "600000.SH"]
    df = provider.get_intraday_latest(count=3)
    assert fake.kline_calls[0][0] == ["000001.SZ", "600000.SH"]
    assert df.height == 2
    provider.get_intraday_latest(count=3)  # 第二次命中缓存, 不重拉维表
    assert len(fake.kline_calls) == 2


def test_full_minute_chunk_failure_keeps_other_chunks(monkeypatch):
    class _Flaky(_FakeClient):
        def kline_1m(self, symbols, count):
            if "600000.SH" in symbols:
                raise tc.TdxError("通达信服务不可用")
            return super().kline_1m(symbols, count)

    syms = [f"{i:06d}.SZ" for i in range(150)] + ["600000.SH"]
    fake = _Flaky({s: _min([DAY]) for s in syms})
    monkeypatch.setattr(
        tp, "tdx_client", SimpleNamespace(TdxClient=lambda **kw: fake, TdxError=tc.TdxError)
    )
    df = TdxProvider().get_intraday_batch(syms, count=240)
    assert df["symbol"].n_unique() == 100  # 第一块成功, 坏块不影响其它块


def test_full_minute_all_requests_fail_returns_empty(monkeypatch):
    provider, _ = _minute_provider(monkeypatch, {}, error=tc.TdxError("连接被拒绝"))
    df = provider.get_intraday_batch(["000001.SZ"], count=240)
    assert df.is_empty()


def test_full_minute_empty_symbols_short_circuits(monkeypatch):
    provider, fake = _minute_provider(monkeypatch, {})
    assert provider.get_intraday_batch([], count=240).is_empty()
    assert fake.kline_calls == []


def test_test_dataset_full_minute(monkeypatch):
    provider, _ = _minute_provider(monkeypatch, {"600519.SH": _min([DAY])})
    out = provider.test_dataset("full_minute", ["600519.SH"])
    assert out["rows"] == 1
    assert out["columns"][1] == "datetime"
    assert out["preview"][0]["datetime"] == "2026-09-21T09:31:00"
    assert out["error"] is None


def test_test_dataset_full_minute_explains_empty(monkeypatch):
    """非交易时段返回空是预期行为, 要给出原因而不是静默 rows=0。"""
    provider, _ = _minute_provider(monkeypatch, {})
    out = provider.test_dataset("full_minute", ["600519.SH"])
    assert out["rows"] == 0 and "当日暂无" in out["error"]


def test_client_kline_1m_request_shape():
    body = {
        "id": 1,
        "result": {"ErrorId": "0", "Value": {"000001.SZ": {"Date": [DAY]}, "bad": "x"}},
    }
    client = _RecordingClient(json.dumps(body).encode())
    out = client.kline_1m(["000001.SZ", "bad"], 3)
    assert out == {"000001.SZ": {"Date": [DAY]}}  # 非 dict 条目丢掉
    assert client.payloads[0]["method"] == "get_market_data"
    assert client.payloads[0]["params"]["period"] == "1m"
    assert client.payloads[0]["params"]["count"] == 3
    assert client.payloads[0]["params"]["dividend_type"] == "none"


def test_client_market_codes():
    body = {
        "id": 1,
        "result": {
            "ErrorId": "0",
            "Value": [{"Code": "000001.SZ", "Name": "平安银行"}, {"Name": "无代码"}],
        },
    }
    assert _RecordingClient(json.dumps(body).encode()).market_codes() == ["000001.SZ"]


# ---- 清单与注册 ----


def test_manifest_declares_datasets():
    from app.data_providers.custom import loader

    manifest = loader.plugin_manifest("tdx")
    assert manifest is not None
    assert manifest["entry"] == "app.plugins.tdx.provider:TdxProvider"
    assert manifest["check"] == "app.plugins.tdx.provider:availability"
    assert set(manifest.get("datasets") or []) == {"depth5", "full_minute"}
    assert manifest.get("runtime") == "none"
    assert not manifest.get("api_key_env")  # 本机服务, 无 Key 概念
