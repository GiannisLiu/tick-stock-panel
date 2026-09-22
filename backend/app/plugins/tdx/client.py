"""通达信本机行情服务的 HTTP JSON-RPC 客户端。

通达信客户端(TdxW.exe)启动并登录后, 在本机 17709 端口暴露 JSON-RPC 服务, `method`
即 tqcenter 的接口名, 返回 `result.ErrorId == "0"` 为成功、数据在 `result.Value`。

这里只走 HTTP, 不 import tqcenter.py、也不碰 TPythClient.dll: 后者要求同进程 ctypes
直连客户端进程(且校验失败会 close 掉连接), 对后端进程是额外耦合。

地址默认 `http://127.0.0.1:17709/`(后端原生跑在 Windows 上时)。后端跑在 Docker 容器里时
容器的 loopback 不是宿主机, 用环境变量覆盖:
  TDX_BASE_URL=http://host.docker.internal:17709/
"""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:17709/"
BASE_URL_ENV = "TDX_BASE_URL"
DEFAULT_TIMEOUT = 10.0

# 探活与试拉用的固定标的(平安银行, 任一交易日都有盘口)
PROBE_SYMBOL = "000001.SZ"


def default_base_url() -> str:
    """每次读环境变量, 便于改 .env 后重载数据源即刻生效。"""
    return (os.environ.get(BASE_URL_ENV) or "").strip() or DEFAULT_BASE_URL


class TdxError(RuntimeError):
    """通达信服务不可用或接口报错(客户端未启动/未登录/ErrorId != 0)。"""


class TdxClient:
    """17709 JSON-RPC 同步客户端。无长连接、无句柄, 方便被 provider 懒加载复用。"""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url or default_base_url()
        self.timeout = timeout

    def close(self) -> None:
        """无连接可关, 保持与其它插件客户端一致的接口形状。"""

    def _post(self, payload: bytes) -> bytes:
        """唯一的网络出口(测试替换点)。"""
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def rpc(self, method: str, **params: object) -> object:
        """调用一个 tqcenter 接口, 返回 result.Value; 失败抛 TdxError。"""
        payload = json.dumps(
            {"id": 1, "method": method, "params": params}, ensure_ascii=False
        ).encode("utf-8")
        try:
            body = json.loads(self._post(payload).decode("utf-8"))
        except (OSError, ValueError) as e:  # 连接失败/超时/非 JSON
            raise TdxError(f"通达信服务不可用({self.base_url}): {e}") from e
        if not isinstance(body, dict):
            raise TdxError(f"通达信接口 {method} 返回格式异常: {type(body).__name__}")
        if body.get("error"):
            raise TdxError(f"通达信接口 {method} 报错: {body['error']}")
        result = body.get("result")
        if not isinstance(result, dict):
            return result
        error_id = result.get("ErrorId")
        if error_id is not None and str(error_id) != "0":
            raise TdxError(f"通达信接口 {method} 失败: {result.get('Error') or error_id}")
        return result.get("Value", result)

    def snapshot(self, stock_code: str) -> dict:
        """单标的实时快照(含五档 Buyp/Buyv/Sellp/Sellv)。"""
        value = self.rpc("get_market_snapshot", stock_code=stock_code)
        return value if isinstance(value, dict) else {}

    def kline_1m(self, symbols: list[str], count: int) -> dict[str, dict]:
        """批量 1 分钟 K 线。

        单请求上限 100 只标的(超出部分静默丢弃), 分块由调用方负责。
        `count` 是"截止现在往前 count 根", 不含日期区间语义。
        """
        value = self.rpc(
            "get_market_data",
            stock_list=list(symbols),
            period="1m",
            count=int(count),
            dividend_type="none",  # 原始价: 复权统一交给项目的因子管道
        )
        if not isinstance(value, dict):
            return {}
        return {k: v for k, v in value.items() if isinstance(v, dict)}

    def market_codes(self) -> list[str]:
        """A 股全市场代码列表(get_stock_list market=5)。"""
        value = self.rpc("get_stock_list", market="5", list_type=1)
        if not isinstance(value, list):
            return []
        return [str(r["Code"]) for r in value if isinstance(r, dict) and r.get("Code")]

    def ping(self) -> bool:
        """探活: 客户端在跑且行情可用才为 True。不抛异常。

        通达信字段是字符串, 未登录时返回 "0.00", 所以按数值判断而不是真值判断。
        """
        try:
            row = self.snapshot(PROBE_SYMBOL)
        except TdxError:
            return False
        return any(_positive(row.get(k)) for k in ("LastClose", "Now"))


def _positive(raw: object) -> bool:
    try:
        return float(raw) > 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
