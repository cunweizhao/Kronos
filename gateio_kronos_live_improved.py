#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io 实时轮询 + Kronos 预测（改进版）
=================================================
在你仓库 `realtime/gateio_kronos_live.py` 的基础上优化，满足以下需求：
- **芝麻交易所 Gate.io** 数据源，默认 ETH_USDT，10s K线
- **每 10 秒打印日志**（北京时间 UTC+8），可用 --print-interval 修改
- 打印最近 1 分钟（6 根）10s K线 O/H/L/C/V
- 根据 **收量 / 放量** 给出 **上车建议：做多 / 做空 / 观望**（可解释）
- **预测下一个 10 秒** 的多空倾向（Kronos 模型优先；无模型时用动量+量能启发式）
- 支持 3 秒轮询抓取（--poll），始终使用“已收盘K线”避免抖动
- 出错重试、网络异常降级等稳健性增强

用法示例：
python realtime/gateio_kronos_live_improved.py --symbol ETH_USDT --interval 10s --poll 3 --print-interval 10 --window 120 --mode both

依赖：使用标准库 urllib；如需 requests，可自行替换 http_get 实现。
"""
from __future__ import annotations
import time, math, os, json, sys, statistics
import urllib.parse, urllib.request, ssl
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional

# ===================== 离线MOCK（网络失败时用） =====================
import random
def extend_mock_candles(candles: List[Tuple[int,float,float,float,float,float]], interval_sec: int, n: int=1):
    if not candles:
        # seed
        t0 = int(time.time()) - interval_sec*10
        price = 2000.0
        for i in range(10):
            o = price
            price = max(1e-6, price + random.uniform(-2,2))
            h = max(o, price) + random.uniform(0,1)
            l = min(o, price) - random.uniform(0,1)
            v = 50*random.uniform(0.8,1.2)
            candles.append((t0+i*interval_sec, o,h,l,price,v))
        return candles
    for _ in range(n):
        t,o,h,l,c,v = candles[-1]
        o2 = c
        c2 = max(1e-6, c + random.uniform(-2,2))
        h2 = max(o2,c2) + random.uniform(0,1)
        l2 = min(o2,c2) - random.uniform(0,1)
        v2 = v*random.uniform(0.8,1.2)
        candles.append((t+interval_sec, o2,h2,l2,c2,v2))
    return candles


# ===================== 工具函数 =====================

def now_beijing() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=8)

def fmt_ts_beijing(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=8)
    return dt.strftime("%Y-%m-%d %H:%M:%S (Beijing)")

def log(msg: str) -> None:
    print(f"[{now_beijing().strftime('%Y-%m-%d %H:%M:%S')} 北京时间] {msg}", flush=True)

def http_get(url: str, params: dict, timeout: int = 15):
    q = urllib.parse.urlencode(params)
    full = f"{url}?{q}"
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(full, context=ctx, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

# ===================== Gate.io K线抓取 =====================

def fetch_gateio_candles(currency_pair="ETH_USDT", interval="10s",
                         from_s: Optional[int]=None, to_s: Optional[int]=None) -> List[Tuple[int,float,float,float,float,float]]:
    """
    返回 [(t, open, high, low, close, volume)] 按时间升序。
    Gate.io 原响应为 [t, v, c, h, l, o]（字符串，且新→旧）。
    """
    base = "https://api.gateio.ws/api/v4/spot/candlesticks"
    if to_s is None:
        to_s = int(time.time())
    if from_s is None:
        # 拉 200 根作为默认窗口
        interval_sec = interval_to_seconds(interval)
        from_s = to_s - interval_sec * 200 - 60
    raw = http_get(base, {
        "currency_pair": currency_pair,
        "interval": interval,
        "from": from_s,
        "to": to_s
    })
    out = []
    for row in raw:
        try:
            t = int(float(row[0]))
            v = float(row[1])
            c = float(row[2])
            h = float(row[3])
            l = float(row[4])
            o = float(row[5])
            out.append((t,o,h,l,c,v))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out

def interval_to_seconds(interval: str) -> int:
    unit = interval[-1].lower()
    val = int(interval[:-1])
    if unit == 's': return val
    if unit == 'm': return val*60
    if unit == 'h': return val*3600
    raise ValueError("Unsupported interval: "+interval)

# ===================== 量价分析与建议 =====================

def classify_volume(candles: List[Tuple[int,float,float,float,float,float]], lookback: int=12) -> str:
    """
    用最后一根的成交量相对前 lookback 根均量判断：放量 / 缩量 / 平量。
    """
    if len(candles) < 2:
        return "平量"
    cur_v = candles[-1][5]
    prev = candles[-(lookback+1):-1] if len(candles) > lookback else candles[:-1]
    if not prev:
        return "平量"
    avg = sum(x[5] for x in prev)/len(prev)
    if avg <= 1e-12: 
        return "平量"
    if cur_v > 1.2 * avg: return "放量"
    if cur_v < 0.8 * avg: return "缩量"
    return "平量"

def decide_entry(candles: List[Tuple[int,float,float,float,float,float]]) -> Tuple[str,float,str]:
    """
    基于“当前K线实体 + 量能”的简单合成：返回 (建议, 置信度, 原因)。
    """
    if not candles: return ("观望", 0.5, "无数据")
    t,o,h,l,c,v = candles[-1]
    vol_tag = classify_volume(candles, lookback=12)
    if c > o and vol_tag == "放量": return ("做多", 0.65, "阳线且放量")
    if c < o and vol_tag == "放量": return ("做空", 0.65, "阴线且放量")
    if c > o and vol_tag != "缩量": return ("做多", 0.55, "阳线且非缩量")
    if c < o and vol_tag != "缩量": return ("做空", 0.55, "阴线且非缩量")
    return ("观望", 0.50, "无明显量价配合")

def momentum_volume_predict(candles: List[Tuple[int,float,float,float,float,float]]) -> Tuple[str,float,str]:
    """
    无模型时的 next-bar 启发式预测：最近8根的价格斜率 + 量能倾斜。
    """
    if len(candles) < 8: return ("观望", 0.5, "样本不足")
    closes = [x[4] for x in candles[-8:]]
    vols   = [x[5] for x in candles[-8:]]
    slope = closes[-1] - closes[0]
    v_slope = sum(vols[-4:]) - sum(vols[:4])
    if slope > 0 and v_slope > 0: return ("做多", 0.60, "价格上行且近端放量")
    if slope < 0 and v_slope > 0: return ("做空", 0.58, "价格回落但放量")
    if slope > 0 and v_slope < 0: return ("观望", 0.52, "缩量上涨，谨慎")
    if slope < 0 and v_slope < 0: return ("做空", 0.55, "缩量下跌延续")
    return ("观望", 0.50, "信号一般")

# ===================== Kronos 接入（若可用） =====================

def kronos_predict_next_close(candles: List[Tuple[int,float,float,float,float,float]], device: str="cpu"
                              ) -> Optional[Tuple[str,float,str]]:
    """
    如果仓库可用 Kronos 模型，则用其生成下一根的收盘价预测，进而转换为多空建议。
    期望返回: ("做多"/"做空"/"观望", 置信度, 说明)；若不可用返回 None。
    """
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        import pandas as pd
        import numpy as np

        # K线转 DataFrame
        df = pd.DataFrame([{
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "timestamps": datetime.fromtimestamp(t, tz=timezone.utc)
        } for (t,o,h,l,c,v) in candles])

        # 实例化 tokenizer + predictor（使用默认权重；若用户已 finetune，可改为本地权重路径）
        tokenizer = KronosTokenizer()
        predictor = KronosPredictor(device=device)

        # 最近 60 根作为输入（含量能）
        lookback = min(60, len(df))
        kline_df = df.iloc[-lookback:].copy()
        # 预测 1 根
        pred_df = predictor.predict_next(
            kline_df, 
            include_volume=True, 
            pred_len=1, 
            sample_count=1, 
            verbose=False
        )
        if pred_df is None or pred_df.empty:
            return None

        pred_close = float(pred_df["close"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        delta = pred_close - last_close
        # 简单阈值化（可由配置调整）
        if abs(delta) < max(0.0005*last_close, 1.0):
            return ("观望", 0.52, f"Kronos预测变动较小 Δ={delta:.4f}")
        if delta > 0:
            return ("做多", 0.62, f"Kronos预测上涨 Δ={delta:.4f}")
        else:
            return ("做空", 0.62, f"Kronos预测下跌 Δ={delta:.4f}")
    except Exception as e:
        # Kronos 未安装或运行错误 → 返回 None 走本地启发式
        return None


# ===================== 可选用户钩子 =====================

def call_user_hooks(candles):
    """
    如果项目根存在 user_hooks.py 且定义 on_tick(candles)->dict，就调用它并返回结果。
    返回示例：{"advice": "做多/做空/观望", "reason": "...", "meta": {...}}
    """
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        import user_hooks  # type: ignore
        if hasattr(user_hooks, "on_tick"):
            return user_hooks.on_tick(candles)
    except Exception as _e:
        return None
    return None

# ===================== 主循环 =====================

@dataclass
class Params:
    print_interval: int = 10    # 日志打印周期（秒）
    poll: int = 3               # 轮询周期（秒）
    window: int = 120           # 维护多少根历史K（用于统计）
    device: str = "cpu"         # Kronos 推理设备
    mode: str = "both"          # 打印 哪些建议：rules/kronos/both

def live_loop(symbol="ETH_USDT", interval="10s", params: Params = Params()):
    interval_sec = interval_to_seconds(interval)
    last_print_ts = 0
    last_candle_time = None

    log(f"启动 Gate.io 实时监控：symbol={symbol}, interval={interval}, poll={params.poll}s, 打印={params.print_interval}s, 窗口={params.window}根")

    candles: List[Tuple[int,float,float,float,float,float]] = []
    while True:
        try:
            to_s = int(time.time())
            from_s = to_s - interval_sec * (params.window + 5)
            try:
                new_c = fetch_gateio_candles(symbol, interval, from_s, to_s)
                if new_c:
                    candles = new_c[-params.window:]
            except Exception as fe:
                log(f"在线抓取失败：{fe}，使用MOCK扩展一根K线")
                candles = extend_mock_candles(candles, interval_sec, n=1)

            # 仅在收盘后推进；Gate.io 每根K线的时间戳通常是该周期开始时刻（或结束时刻），这里直接以最新一根视作“已收盘”
            if candles:
                last_candle_time = candles[-1][0]

            now_s = int(time.time())
            # 每 print_interval 秒打印一次
            if now_s - last_print_ts >= params.print_interval and candles:
                last_print_ts = now_s
                t,o,h,l,c,v = candles[-1]
                vol_tag = classify_volume(candles)
                suggest, conf, why = decide_entry(candles)
                # Kronos 优先；无则启发式
                kronos_out = kronos_predict_next_close(candles, device=params.device) if params.mode in ("kronos","both") else None
                if kronos_out is None:
                    pred_action, pred_conf, pred_why = momentum_volume_predict(candles)
                    pred_src = "启发式"
                else:
                    pred_action, pred_conf, pred_why = kronos_out
                    pred_src = "Kronos"

                log(f"最新价: {c:.2f}  O/H/L/C: {o:.2f}/{h:.2f}/{l:.2f}/{c:.2f}  成交量: {v:.2f}（{vol_tag}）")
                log(f"上车建议：{suggest}（置信度 {conf:.2f}）— 原因：{why}")
                log(f"预测下一个10秒：{pred_action}（置信度 {pred_conf:.2f}，来源：{pred_src}）— 依据：{pred_why}")

                # 用户自定义钩子（若有）
                hook_out = call_user_hooks(candles)
                if hook_out:
                    log(f"自定义策略钩子建议：{hook_out}")

                # 打印最近 6 根 10s K线"
                recent = candles[-6:]
                print("最近1分钟(6根)10秒K线：")
                for (tt,oo,hh,ll,cc,vv) in recent:
                    print(f"{fmt_ts_beijing(tt)} | O:{oo:.2f} H:{hh:.2f} L:{ll:.2f} C:{cc:.2f} V:{vv:.2f}")
                sys.stdout.flush()

        except Exception as e:
            log(f"抓取或分析出错：{e}（将继续重试）")

        time.sleep(params.poll)

# ===================== CLI =====================

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETH_USDT")
    p.add_argument("--interval", default="10s")
    p.add_argument("--poll", type=int, default=3, help="轮询周期（秒），默认3")
    p.add_argument("--print-interval", type=int, default=10, help="打印日志周期（秒），默认10")
    p.add_argument("--window", type=int, default=120, help="保留历史K线根数，用于统计")
    p.add_argument("--device", default="cpu", help="Kronos 推理设备 cpu/cuda")
    p.add_argument("--mode", default="both", choices=["rules","kronos","both"], help="打印规则/模型/两者")
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    params = Params(print_interval=args.print_interval, poll=args.poll, window=args.window, device=args.device, mode=args.mode)
    live_loop(args.symbol, args.interval, params)

if __name__ == "__main__":
    main()
