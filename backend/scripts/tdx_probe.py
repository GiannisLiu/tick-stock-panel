#!/usr/bin/env python3
"""通达信(tdx)数据源探针: 判定五档盘口与全量分钟能不能真正投用。

默认 dry-run(不联网)。--live 才连本机通达信行情服务(17709)逐项实测:

  1. connectivity      服务可达 + 行情可用(客户端已启动并登录)
  2. minute_coverage   全市场当日 1 分钟线覆盖(客户端是按需补数据, 冷缓存要连跑几遍才收敛)
  3. minute_freshness  末根 bar 距现在多久(盘中核心指标: 分钟级延迟能不能接受)
  4. minute_bars       单只当日根数(盘中应随分钟增长)
  5. depth5            抽样标的的盘口档数与单次耗时(L2 权限决定能不能给满五档)
  6. increment_round   全市场增量轮(count=3)耗时(minute_refresh 稳态轮基准)

退出码: 0 无 FAIL(允许 WARN) / 1 有 FAIL / 2 客户端构造失败。

用法:
  cd backend && uv run python scripts/tdx_probe.py                    # dry-run
  cd backend && uv run python scripts/tdx_probe.py --live
  cd backend && uv run python scripts/tdx_probe.py --live --passes 3  # 看冷缓存收敛
  docker exec tsp /app/.venv/bin/python /app/scripts/tdx_probe.py --live

报告: data/reports/tdx_probe/<时间戳>/probe_summary.json
  (默认落数据目录, 容器内为 /app/data/reports/..., 映射回宿主机的 data/reports/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Protocol

# 既支持 `cd backend && uv run python scripts/tdx_probe.py`, 也支持从仓库根目录或
# 容器内直接跑: 先把 backend/ 与仓库根放进 sys.path, 再 import app。
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "backend"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.market_time import CN_TZ, cn_now, cn_today, in_continuous_session  # noqa: E402
from app.plugins.tdx.client import TdxClient, default_base_url  # noqa: E402

# 判定阈值(实测基准: 全市场当日覆盖 5565/5575, 增量轮 2.9s, 快照 12.7ms/只)
COVERAGE_OK = 0.95
COVERAGE_WARN = 0.80
LAG_OK_S = 180.0  # 3 分钟: 与 minute_refresh 的修复轮阈值一致
LAG_WARN_S = 600.0
ROUND_OK_S = 10.0
BARS_OK_RATIO = 0.9
DEFAULT_PASSES = 1
DEFAULT_SAMPLE = 30
PROBE_SYMBOL = "000001.SZ"
DAY_BARS = 240  # A 股每交易日 1 分钟根数(9:31-15:00)
_BATCH = 100  # get_market_data 单请求标的数上限


class TdxLike(Protocol):
    """探针只依赖这四个方法, 测试用假客户端注入。"""

    def ping(self) -> bool: ...

    def market_codes(self) -> list[str]: ...

    def kline_1m(self, symbols: list[str], count: int) -> dict[str, dict]: ...

    def snapshot(self, stock_code: str) -> dict: ...


@dataclass
class Check:
    name: str
    status: str  # ok / warn / fail
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    status: str  # ok / warn / fail / dry_run
    generated_at: str
    base_url: str
    passes: int
    checks: list[Check]


def _repo_root() -> Path:
    return _REPO_ROOT


def _chunks(items: list[str], size: int = _BATCH) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _num(raw: object) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _naive_cn(moment: datetime) -> datetime:
    """统一成北京墙钟 naive: cn_now() 返回 aware, 测试注入的是 naive。"""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(CN_TZ).replace(tzinfo=None)


def _bar_dt(day: object, clock: object) -> datetime | None:
    """通达信的 Date(20260921) + Time(93100) → 北京墙钟 datetime。"""
    try:
        ts = str(clock).zfill(6)
        return datetime(
            int(str(day)[0:4]),
            int(str(day)[4:6]),
            int(str(day)[6:8]),
            int(ts[0:2]),
            int(ts[2:4]),
            int(ts[4:6]),
        )
    except (ValueError, IndexError):
        return None


def _today_bars(row: dict, day: str) -> int:
    return sum(1 for d in (row.get("Date") or []) if str(d) == day)


def _minutes_since(start: dt_time, now: dt_time) -> int:
    return (now.hour * 60 + now.minute) - (start.hour * 60 + start.minute) + 1


def expected_bars(now: datetime) -> int | None:
    """当日到 now 为止应有的 1 分钟根数; 盘前/非交易日返回 None(不判定)。"""
    if now.weekday() >= 5:
        return None
    t = now.time()
    if t < dt_time(9, 31):
        return None
    if t <= dt_time(11, 30):
        return _minutes_since(dt_time(9, 31), t)
    if t < dt_time(13, 1):
        return 120
    if t <= dt_time(15, 0):
        return 120 + _minutes_since(dt_time(13, 1), t)
    return DAY_BARS


# ---------------------------------------------------------------- 检查项


def _check_connectivity(client: TdxLike) -> Check:
    t0 = time.perf_counter()
    try:
        ok = bool(client.ping())
    except Exception as e:  # 探测本身失败也要给结论, 不把异常抛给调用方
        return Check("connectivity", "fail", f"探测异常: {e}")
    ms = (time.perf_counter() - t0) * 1000
    if not ok:
        return Check(
            "connectivity",
            "fail",
            "通达信服务不可达或未登录(客户端没启动/没登录, 或地址不对)",
        )
    return Check("connectivity", "ok", f"服务可用 ({ms:.0f}ms)", {"ms": round(ms, 1)})


def _worse(a: str, b: str) -> str:
    """取两个状优中更差的(ok < warn < fail)。"""
    order = {"ok": 0, "warn": 1, "fail": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _check_minute_coverage(client: TdxLike, codes: list[str], passes: int) -> Check:
    if not codes:
        return Check("minute_coverage", "fail", "取不到全市场代码列表")
    day = cn_today().strftime("%Y%m%d")
    chunks = _chunks(codes)
    history: list[dict] = []
    for i in range(max(1, passes)):
        cnt: Counter = Counter()
        failed = 0
        seen = 0
        t0 = time.perf_counter()
        for chunk in chunks:
            try:
                data = client.kline_1m(chunk, 1)
            except Exception:
                # 实测通达信服务在连续请求下偶发掉连接: 单块失败不能算整项失败,
                # 否则探针会把瞬时抖动报成"分钟数据不可用"
                failed += 1
                continue
            seen += len(data)
            for row in data.values():
                dates = row.get("Date") or []
                if dates:
                    cnt[str(dates[-1])] += 1
        history.append(
            {
                "pass": i + 1,
                "today": cnt.get(day, 0),
                "symbols": seen,
                "failed_chunks": failed,
                "secs": round(time.perf_counter() - t0, 2),
            }
        )
    last = history[-1]
    if last["failed_chunks"] >= len(chunks):
        return Check(
            "minute_coverage", "fail", f"全部 {len(chunks)} 块取数失败", {"passes": history}
        )
    fetched = last["symbols"]
    if fetched == 0:
        return Check("minute_coverage", "fail", "接口返回空数据", {"passes": history})
    # 覆盖率按"真正读到"的标的口径算: 掉块属网络问题, 不该假装成数据覆盖不足
    today = last["today"]
    ratio = today / fetched
    status = "ok" if ratio >= COVERAGE_OK else "warn" if ratio >= COVERAGE_WARN else "fail"
    if last["failed_chunks"]:
        status = _worse(status, "warn")
    detail = f"当日覆盖 {today}/{fetched} = {ratio:.1%}"
    if len(history) > 1:
        detail += f" (共 {len(history)} 遍, 首遍 {history[0]['today']})"
    if last["failed_chunks"]:
        detail += f", {last['failed_chunks']}/{len(chunks)} 块取数失败(共请求 {len(codes)} 只)"
    return Check("minute_coverage", status, detail, {"passes": history, "ratio": round(ratio, 4)})


def _check_minute_freshness(client: TdxLike, now: datetime) -> Check:
    now = _naive_cn(now)
    try:
        data = client.kline_1m([PROBE_SYMBOL], 3)
    except Exception as e:
        return Check("minute_freshness", "fail", f"取数失败: {e}")
    row = data.get(PROBE_SYMBOL) or {}
    dates, times = row.get("Date") or [], row.get("Time") or []
    if not dates or not times:
        return Check("minute_freshness", "fail", f"{PROBE_SYMBOL} 无分钟数据")
    last = _bar_dt(dates[-1], times[-1])
    if last is None:
        return Check("minute_freshness", "fail", "末根时间字段无法解析")
    lag = (now - last).total_seconds()
    payload = {"last_bar": last.strftime("%Y-%m-%d %H:%M"), "lag_seconds": round(lag)}
    if not in_continuous_session(now):
        return Check("minute_freshness", "ok", f"末根 {last:%H:%M}, 非交易时段不判定延迟", payload)
    status = "ok" if lag <= LAG_OK_S else "warn" if lag <= LAG_WARN_S else "fail"
    return Check(
        "minute_freshness", status, f"末根 {last:%H:%M}, 延迟 {lag / 60:.1f} 分钟", payload
    )


def _check_minute_bars(client: TdxLike, now: datetime) -> Check:
    now = _naive_cn(now)
    day = now.strftime("%Y%m%d")
    try:
        data = client.kline_1m([PROBE_SYMBOL], DAY_BARS + 1)
    except Exception as e:
        return Check("minute_bars", "fail", f"取数失败: {e}")
    count = _today_bars(data.get(PROBE_SYMBOL) or {}, day)
    want = expected_bars(now)
    payload = {"symbol": PROBE_SYMBOL, "bars": count, "expected": want}
    if want is None:
        return Check(
            "minute_bars", "ok", f"{PROBE_SYMBOL} 当日 {count} 根(盘前/非交易日不判定)", payload
        )
    if count >= want * BARS_OK_RATIO:
        return Check(
            "minute_bars", "ok", f"{PROBE_SYMBOL} 当日 {count} 根, 应有 {want} 根", payload
        )
    status = "warn" if count >= want * 0.5 else "fail"
    return Check(
        "minute_bars", status, f"{PROBE_SYMBOL} 当日 {count} 根, 应有 {want} 根(偏少)", payload
    )


def _check_depth(client: TdxLike, sample: list[str]) -> Check:
    if not sample:
        return Check("depth5", "warn", "无抽样标的")
    levels: Counter = Counter()
    failed = 0
    t0 = time.perf_counter()
    for sym in sample:
        try:
            row = client.snapshot(sym)
        except Exception:
            failed += 1
            continue
        levels[sum(1 for p in (row.get("Buyp") or []) if _num(p) > 0)] += 1
    ms = (time.perf_counter() - t0) * 1000 / len(sample)
    covered = sum(v for k, v in levels.items() if k > 0)
    payload = {
        "sample": len(sample),
        "covered": covered,
        "levels": {str(k): v for k, v in sorted(levels.items())},
        "ms_per_symbol": round(ms, 2),
    }
    if covered == 0:
        return Check("depth5", "fail", f"抽样 {len(sample)} 只全部无盘口", payload)
    ratio = covered / len(sample)
    status = "ok" if ratio >= BARS_OK_RATIO else "warn"
    note = f", {failed} 只取数失败" if failed else ""
    return Check(
        "depth5",
        status,
        f"抽样 {len(sample)} 只有盘口 {covered} 只, 档数分布 {sorted(levels.items())}, {ms:.1f}ms/只{note}",
        payload,
    )


def _check_increment_round(client: TdxLike, codes: list[str]) -> Check:
    if not codes:
        return Check("increment_round", "warn", "无全市场代码, 跳过")
    chunks = _chunks(codes)
    failed = 0
    t0 = time.perf_counter()
    for chunk in chunks:
        try:
            client.kline_1m(chunk, 3)
        except Exception:
            failed += 1  # 与覆盖率同理: 单块抖动不当作整轮不可用
    secs = time.perf_counter() - t0
    if failed >= len(chunks):
        return Check("increment_round", "fail", f"全部 {len(chunks)} 块取数失败")
    status = "ok" if secs <= ROUND_OK_S else "warn"
    detail = f"全市场 count=3 一轮 {secs:.2f}s (minute_refresh 稳态轮基准)"
    if failed:
        status = _worse(status, "warn")
        detail += f", {failed}/{len(chunks)} 块取数失败"
    return Check(
        "increment_round", status, detail, {"seconds": round(secs, 2), "failed_chunks": failed}
    )


# ---------------------------------------------------------------- 编排


def report_status(checks: list[Check]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" for c in checks) else 0


def run_live(
    client: TdxLike,
    *,
    passes: int = DEFAULT_PASSES,
    sample: int = DEFAULT_SAMPLE,
    now: datetime | None = None,
) -> Report:
    """真探测一遍(now 可注入, 便于测试时段相关判定)。"""
    now = _naive_cn(now or cn_now())
    checks = [_check_connectivity(client)]
    if checks[0].status != "fail":  # 连不上就没必要往下测
        codes = client.market_codes()
        checks.append(_check_minute_coverage(client, codes, passes))
        checks.append(_check_minute_freshness(client, now))
        checks.append(_check_minute_bars(client, now))
        checks.append(_check_depth(client, codes[: max(1, sample)]))
        checks.append(_check_increment_round(client, codes))
    return Report(
        status=report_status(checks),
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S"),
        base_url=default_base_url(),
        passes=passes,
        checks=checks,
    )


def run_dry_run(*, passes: int = DEFAULT_PASSES, sample: int = DEFAULT_SAMPLE) -> Report:
    """不联网: 只确认客户端配置与待检查项。"""
    plan = [
        ("connectivity", "探测 17709 服务可达性"),
        ("minute_coverage", f"全市场当日 1m 覆盖({passes} 遍)"),
        ("minute_freshness", "末根 bar 延迟"),
        ("minute_bars", "单只当日根数"),
        ("depth5", f"盘口档数与耗时(抽样 {sample} 只)"),
        ("increment_round", "全市场增量轮耗时"),
    ]
    return Report(
        status="dry_run",
        generated_at=_naive_cn(cn_now()).strftime("%Y-%m-%d %H:%M:%S"),
        base_url=default_base_url(),
        passes=passes,
        checks=[Check(name, "dry_run", detail) for name, detail in plan],
    )


def _default_out_dir(stamp: str) -> Path:
    """默认报告目录: 落数据目录而不是仓库根 - 容器内根目录不可写且写入即丢,
    落数据目录后宿主机能从 data/reports/ 看到同一份报告。"""
    from app.config import settings

    return Path(settings.data_dir) / "reports" / "tdx_probe" / stamp


def _write_report(report: Report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "probe_summary.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _print_report(report: Report, path: Path) -> None:
    print(
        f"通达信数据源探针  base_url={report.base_url}  现在 {report.generated_at}  模式={report.status}"
    )
    for c in report.checks:
        print(f"[{c.status.upper():>8}] {c.name:<16} {c.detail}")
    n_ok = sum(1 for c in report.checks if c.status == "ok")
    n_warn = sum(1 for c in report.checks if c.status == "warn")
    n_fail = sum(1 for c in report.checks if c.status == "fail")
    print(f"结论: {report.status.upper()} ({n_ok} OK / {n_warn} WARN / {n_fail} FAIL)")
    print(f"报告: {path}")


def main(argv: list[str] | None = None, client: TdxLike | None = None) -> int:
    parser = argparse.ArgumentParser(description="通达信数据源探针(默认 dry-run, --live 才真探测)")
    parser.add_argument("--live", action="store_true", help="真连本机通达信 17709")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="全市场覆盖连测几遍")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="五档抽样标的数")
    parser.add_argument(
        "--out", type=Path, default=None, help="报告目录(默认 data/reports/tdx_probe/<时间戳>)"
    )
    args = parser.parse_args(argv)

    stamp = _naive_cn(cn_now()).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or _default_out_dir(stamp)

    if not args.live:
        report = run_dry_run(passes=args.passes, sample=args.sample)
    else:
        if client is None:
            try:
                client = TdxClient()
            except Exception as e:
                print(f"[FAIL] 客户端构造失败: {e}")
                return 2
        report = run_live(client, passes=args.passes, sample=args.sample)

    path = _write_report(report, out_dir)
    _print_report(report, path)
    return 0 if not args.live else exit_code(report.checks)


if __name__ == "__main__":
    sys.exit(main())
