"""tdx_probe 判定逻辑测试(不联网: 假客户端 + now 注入)。

覆盖: dry-run 不发请求、覆盖率三档判定与冷缓存收敛、末根延迟在/不在交易时段、
当日根数期望值、盘口档数判定、退出码语义、报告落盘。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scripts.tdx_probe import (
    Check,
    _check_connectivity,
    _check_depth,
    _check_increment_round,
    _check_minute_bars,
    _check_minute_coverage,
    _check_minute_freshness,
    _default_out_dir,
    exit_code,
    expected_bars,
    main,
    report_status,
    run_dry_run,
    run_live,
)

DAY = "20260921"  # 周一
OLD = "20260915"
NOW_SESSION = datetime(2026, 9, 21, 10, 30)
NOW_CLOSED = datetime(2026, 9, 21, 22, 0)


class _Fake:
    """dates 按标的顺序循环决定"最新日期"; times 循环决定 bar 时刻。

    depth_levels 决定盘口非零档数, bars_per_day 决定单只当日根数。
    """

    def __init__(
        self,
        *,
        dates=None,
        times=None,
        codes=None,
        depth_levels=1,
        bars_per_day=1,
        error=None,
    ):
        self.dates = list(dates) if dates else [DAY]
        self.times = list(times) if times else ["150000"]
        self.codes = list(codes) if codes else []
        self.depth_levels = depth_levels
        self.bars_per_day = bars_per_day
        self.error = error
        self.calls: list[str] = []
        self._i = 0

    def ping(self) -> bool:
        self.calls.append("ping")
        if self.error:
            raise self.error
        return True

    def market_codes(self) -> list[str]:
        self.calls.append("market_codes")
        return list(self.codes)

    def kline_1m(self, symbols, count):
        self.calls.append(f"kline:{len(symbols)}:{count}")
        if self.error:
            raise self.error
        out = {}
        for s in symbols:
            day = self.dates[self._i % len(self.dates)]
            self._i += 1
            n = min(self.bars_per_day, count)
            out[s] = {
                "Date": [day] * n,
                "Time": [self.times[j % len(self.times)] for j in range(n)],
                "Amount": [1.0] * n,
            }
        return out

    def snapshot(self, stock_code):
        self.calls.append(f"snapshot:{stock_code}")
        prices = [f"{10 + i:.2f}" if i < self.depth_levels else "0.00" for i in range(5)]
        vols = ["1" if i < self.depth_levels else "0" for i in range(5)]
        return {"Buyp": prices, "Buyv": vols, "Sellp": prices, "Sellv": vols, "Now": "10.00"}


def _codes(n: int) -> list[str]:
    return [f"{i:06d}.SZ" for i in range(n)]


# ---- dry-run 与退出码 ----


def test_dry_run_makes_no_calls(tmp_path):
    fake = _Fake(error=AssertionError("dry-run 不该发任何请求"))
    assert main(["--out", str(tmp_path)], client=fake) == 0
    assert fake.calls == []
    report = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert next(c["name"] for c in report["checks"]) == "connectivity"


def test_default_out_dir_under_data_dir():
    """默认报告落数据目录: 容器内同样可写, 宿主机也能从 data/reports/ 看到。"""
    from app.config import settings

    path = _default_out_dir("20260921-220000")
    assert path.parent.parent.parent == Path(settings.data_dir)
    assert path.name == "20260921-220000"


def test_report_status_and_exit_code():
    assert report_status([Check("a", "ok", ""), Check("b", "warn", "")]) == "warn"
    assert report_status([Check("a", "ok", ""), Check("b", "fail", "")]) == "fail"
    assert exit_code([Check("a", "warn", "")]) == 0  # WARN 不阻断
    assert exit_code([Check("a", "fail", "")]) == 1


def test_connectivity_fail_message():
    class _Dead(_Fake):
        def ping(self):
            return False

    check = _check_connectivity(_Dead())
    assert check.status == "fail" and "未登录" in check.detail


# ---- 覆盖率 ----


def test_coverage_ok():
    codes = _codes(100)
    check = _check_minute_coverage(_Fake(codes=codes), codes, 1)
    assert check.status == "ok" and "100.0%" in check.detail


def test_coverage_warn_at_85_percent():
    codes = _codes(20)
    dates = [DAY] * 17 + [OLD] * 3  # 17/20 = 85%
    check = _check_minute_coverage(_Fake(dates=dates, codes=codes), codes, 1)
    assert check.status == "warn" and check.data["ratio"] == 0.85


def test_coverage_fail_when_no_today_bars():
    codes = _codes(20)
    check = _check_minute_coverage(_Fake(dates=[OLD], codes=codes), codes, 1)
    assert check.status == "fail" and check.data["ratio"] == 0.0


def test_coverage_reports_cold_cache_convergence():
    codes = _codes(10)

    class _Warming(_Fake):
        def __init__(self):
            super().__init__(codes=codes)
            self.round = 0

        def kline_1m(self, symbols, count):
            self.round += 1
            day = DAY if self.round >= 3 else OLD  # 冷缓存: 前两遍给历史 bar
            return {s: {"Date": [day]} for s in symbols}

    check = _check_minute_coverage(_Warming(), codes, 3)
    assert check.status == "ok"
    assert [h["today"] for h in check.data["passes"]] == [0, 0, 10]


def test_coverage_fails_on_client_error():
    """全部块都失败才是 FAIL(单块抖动只降级, 见下一个用例)。"""
    check = _check_minute_coverage(_Fake(error=RuntimeError("connection refused")), _codes(1), 1)
    assert check.status == "fail" and "全部" in check.detail


def test_coverage_survives_partial_chunk_failure():
    """实测通达信连续请求会偶发掉连接: 掉一块只算 WARN, 剩下的数据照收。"""
    codes = _codes(250)  # 3 块

    class _Flaky(_Fake):
        def __init__(self):
            super().__init__(codes=codes)
            self.n = 0

        def kline_1m(self, symbols, count):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("Network is unreachable")
            return {s: {"Date": [DAY]} for s in symbols}

    check = _check_minute_coverage(_Flaky(), codes, 1)
    assert check.status == "warn"
    assert "1/3 块取数失败" in check.detail
    assert check.data["passes"][0]["failed_chunks"] == 1


def test_increment_round_tolerates_partial_failure():
    codes = _codes(250)

    class _Flaky(_Fake):
        def __init__(self):
            super().__init__(codes=codes)
            self.n = 0

        def kline_1m(self, symbols, count):
            self.n += 1
            if self.n <= 2:
                raise RuntimeError("boom")
            return {s: {"Date": [DAY]} for s in symbols}

    check = _check_increment_round(_Flaky(), codes)
    assert check.status == "warn" and check.data["failed_chunks"] == 2
    dead = _check_increment_round(_Fake(error=RuntimeError("boom")), codes)
    assert dead.status == "fail"


# ---- 末根延迟 ----


def test_freshness_ok_in_session():
    check = _check_minute_freshness(_Fake(times=["102900"]), NOW_SESSION)
    assert check.status == "ok" and check.data["lag_seconds"] == 60


def test_freshness_warn_then_fail_in_session():
    warn = _check_minute_freshness(_Fake(times=["102000"]), NOW_SESSION)  # 10 分钟
    assert warn.status == "warn"
    fail = _check_minute_freshness(_Fake(times=["093500"]), NOW_SESSION)  # 55 分钟
    assert fail.status == "fail" and "55.0 分钟" in fail.detail


def test_freshness_not_judged_out_of_session():
    check = _check_minute_freshness(_Fake(), NOW_CLOSED)
    assert check.status == "ok" and "非交易时段" in check.detail


def test_freshness_fails_without_data():
    check = _check_minute_freshness(_Fake(bars_per_day=0), NOW_SESSION)
    assert check.status == "fail"


# ---- 当日根数期望值 ----


def test_expected_bars_by_time():
    d = lambda h, m: datetime(2026, 9, 21, h, m)  # noqa: E731
    assert expected_bars(d(9, 0)) is None  # 盘前
    assert expected_bars(d(10, 0)) == 30
    assert expected_bars(d(11, 30)) == 120
    assert expected_bars(d(12, 0)) == 120  # 午休
    assert expected_bars(d(14, 0)) == 180
    assert expected_bars(d(16, 0)) == 240  # 收盘后
    assert expected_bars(datetime(2026, 9, 26, 10, 0)) is None  # 周六


def test_minute_bars_judgement():
    ok = _check_minute_bars(_Fake(bars_per_day=30), datetime(2026, 9, 21, 10, 0))
    assert ok.status == "ok" and ok.data == {"symbol": "000001.SZ", "bars": 30, "expected": 30}

    short = _check_minute_bars(_Fake(bars_per_day=5), datetime(2026, 9, 21, 10, 0))
    assert short.status == "fail"

    pre_open = _check_minute_bars(_Fake(bars_per_day=0), datetime(2026, 9, 21, 9, 0))
    assert pre_open.status == "ok"  # 盘前不判定


# ---- 盘口档数 ----


def test_depth_one_level_is_enough():
    check = _check_depth(_Fake(depth_levels=1), ["000001.SZ", "600000.SH"])
    assert check.status == "ok" and check.data["levels"] == {"1": 2}


def test_depth_full_five_levels():
    check = _check_depth(_Fake(depth_levels=5), ["000001.SZ"])
    assert check.status == "ok" and check.data["levels"] == {"5": 1}


def test_depth_fails_when_all_empty():
    check = _check_depth(_Fake(depth_levels=0), ["000001.SZ"])
    assert check.status == "fail"


# ---- 编排 ----


def test_run_live_skips_remaining_checks_when_unreachable():
    class _Dead(_Fake):
        def ping(self):
            return False

    report = run_live(_Dead(), passes=1, sample=1, now=NOW_CLOSED)
    assert report.status == "fail"
    assert [c.name for c in report.checks] == ["connectivity"]


def test_run_live_full_path_and_dry_run():
    # 收盘后当日应满 240 根: 用足量 1 分钟 bar 的假客户端
    report = run_live(_Fake(codes=_codes(5), bars_per_day=240), passes=1, sample=3, now=NOW_CLOSED)
    assert [c.name for c in report.checks] == [
        "connectivity",
        "minute_coverage",
        "minute_freshness",
        "minute_bars",
        "depth5",
        "increment_round",
    ]
    assert report.status in ("ok", "warn")
    assert run_dry_run().status == "dry_run"


def test_main_live_exit_code_on_failure(tmp_path):
    class _Dead(_Fake):
        def ping(self):
            return False

    assert main(["--live", "--out", str(tmp_path)], client=_Dead()) == 1
    healthy = _Fake(codes=_codes(3), bars_per_day=240)  # 收盘后当日 240 根才算健康
    assert main(["--live", "--out", str(tmp_path)], client=healthy) == 0
