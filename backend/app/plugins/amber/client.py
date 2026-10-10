"""amber 行情服务 HTTP 客户端。

职责: Bearer 认证、错误信封解包、日K/批量分钟/标的清单请求。不知道 provider / services 层。
服务为 FastAPI (自部署, 默认 http://127.0.0.1:8310): 业务错误走 HTTP 状态码 +
{"detail": "..."} 信封 (401 missing bearer key / 400 参数 / 503 数据未就绪)。

响应形态 (源码实测, 2026-10):
- POST /v1/daily {codes, start, end} → {codes, dates, n, open/high/low/close/
  volume/amount: dates×codes 矩阵 (matrix[d][c], 停牌缺失为 null), quality, truncated}
  服务端单响应 cell 上限 400k, 超限从最早日期截断并置 truncated。
- POST /v1/minutes {codes|market+type, start, end, count} → {n, total,
  bars: {code: [{ts, o, h, l, c, v, a}]}, truncated}
  ts 为北京墙钟 ISO 串; count 为每码最近根数上限(≤24000); 响应总 bar 上限 200k。
  codes 单请求 ≤2000 只; B 股不提供服务(服务端剔除)。
- GET /v1/instruments?market=&type= → {n, items: [{code, name, market, type}]}

时间字段口径: ts/day 均为北京时间墙钟(Asia/Shanghai), 代码格式 600519.SH 与项目一致。
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8310"

# 服务端护栏 (amber src/api/market.py): 单响应 bar 总上限 200k / 单请求 codes ≤2000。
# 调用方按此分块; 客户端不做二次拆分。
MINUTES_BAR_CAP = 200_000
MINUTES_CODES_MAX = 2000
MINUTES_COUNT_MAX = 24_000


class AmberError(Exception):
    """amber 接口错误(配置缺失 / 网络失败 / HTTP 状态码非 200)。"""


class AmberClient:
    """amber REST 客户端 (线程安全: httpx.Client 可并发复用)。"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0) -> None:
        if not api_key:
            raise AmberError("未配置 AMBER_API_KEY")
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._http.close()

    # ---- 内部 ----
    def _request(
        self, method: str, path: str, *,
        json_body: dict | None = None, params: dict | None = None,
    ) -> dict:
        """统一请求 + 错误解包。非 200 抛 AmberError(含 detail 文案)。"""
        try:
            resp = self._http.request(method, path, json=json_body, params=params)
        except httpx.HTTPError as e:
            raise AmberError(f"网络请求失败: {e}") from e
        if resp.status_code != 200:
            try:
                detail = (resp.json() or {}).get("detail") or ""
            except ValueError:
                detail = ""
            raise AmberError(f"HTTP {resp.status_code}: {detail} ({path})")
        try:
            payload = resp.json()
        except ValueError as e:
            raise AmberError(f"响应不是 JSON: {path}") from e
        if not isinstance(payload, dict):
            raise AmberError(f"响应结构异常(非对象): {path}")
        return payload

    # ---- 日K ----
    def daily(self, codes: list[str], start: str | None = None, end: str | None = None) -> dict:
        """批量日K (POST /v1/daily)。start/end 为 ISO 日期串(YYYY-MM-DD)。"""
        body: dict = {"codes": codes}
        if start:
            body["start"] = start
        if end:
            body["end"] = end
        return self._request("POST", "/v1/daily", json_body=body)

    # ---- 分钟K ----
    def minutes(
        self,
        codes: list[str] | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        count: int = 240,
        market: str | None = None,
        type_: str | None = None,
    ) -> dict:
        """批量分钟K (POST /v1/minutes)。codes 与 market/type 二选一。

        count 为每码最近根数上限; start/end 为 ISO 日期时间串(北京墙钟)。
        """
        body: dict = {"count": max(1, min(int(count), MINUTES_COUNT_MAX))}
        if codes:
            body["codes"] = codes[:MINUTES_CODES_MAX]
        if start:
            body["start"] = start
        if end:
            body["end"] = end
        if market:
            body["market"] = market
        if type_:
            body["type"] = type_
        return self._request("POST", "/v1/minutes", json_body=body)

    # ---- 实时快照 ----
    def snapshot(self, market: str | None = None, type_: str | None = None) -> dict:
        """全市场最新快照 (GET /v1/snapshot)。列式矩阵, 不传过滤参数即全池。

        形态: {v, day, stale, n, c: [code...], pre/last/o/h/l/vol/amt/b1/bv1/
        a1/av1/in/out: 与 c 等长的列数组}。服务端已剔除 B 股; b1 等旧格式缺列
        由服务端补 0。无逐行时间戳 — 行情归属时间由 provider 用本地时刻兜底。
        """
        params = {k: v for k, v in {"market": market, "type": type_}.items() if v}
        return self._request("GET", "/v1/snapshot", params=params or None)

    # ---- 标的清单 ----
    def instruments(self, market: str | None = None, type_: str | None = None) -> dict:
        """标的清单 (GET /v1/instruments)。也可用作 Key 探活。"""
        params = {k: v for k, v in {"market": market, "type": type_}.items() if v}
        return self._request("GET", "/v1/instruments", params=params or None)
