#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：定位为什么 gateio_kronos_live_improved.py 没有成功运行
用法：
  python diagnose_env.py
"""

import os, sys, ssl, socket, time, json, traceback
from datetime import datetime, timezone, timedelta
import urllib.request, urllib.parse

def now_beijing():
    return datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=8)

def log(msg):
    print(f"[{now_beijing().strftime('%Y-%m-%d %H:%M:%S')} 北京时间] {msg}", flush=True)

def try_http(url, params=None, timeout=10, insecure=False):
    try:
        if params:
            q = urllib.parse.urlencode(params)
            url = f"{url}?{q}"
        ctx = ssl.create_default_context()
        if insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx, timeout=timeout) as resp:
            data = resp.read()
            return True, f"HTTP {resp.status}, {len(data)} bytes"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main():
    log("开始环境诊断")
    log(f"Python 版本: {sys.version.replace(os.linesep,' ')}")
    log(f"工作目录: {os.getcwd()}")
    log(f"文件存在: realtime/gateio_kronos_live_improved.py = {os.path.exists('realtime/gateio_kronos_live_improved.py')}")
    log(f"文件存在: ../user_hooks.py = {os.path.exists(os.path.join('..','user_hooks.py'))}")
    # DNS & 网络
    try:
        ip = socket.gethostbyname("api.gateio.ws")
        log(f"DNS 解析 api.gateio.ws -> {ip}")
    except Exception as e:
        log(f"DNS 解析失败: {e}")

    ok, msg = try_http("https://api.gateio.ws/api/v4/spot/candlesticks", {
        "currency_pair":"ETH_USDT","interval":"10s","from": int(time.time())-1200, "to": int(time.time())
    }, timeout=12, insecure=False)
    log(f"HTTPS 测试（严格验证）: {ok} - {msg}")

    if not ok:
        ok2, msg2 = try_http("https://api.gateio.ws/api/v4/spot/candlesticks", {
            "currency_pair":"ETH_USDT","interval":"10s","from": int(time.time())-1200, "to": int(time.time())
        }, timeout=12, insecure=True)
        log(f"HTTPS 测试（不校验证书，仅排错用）: {ok2} - {msg2}")

    # 试跑改进脚本（MOCK 2 次，验证打印）
    log("尝试以 MOCK 模式快速运行改进脚本（2 次打印）")
    try:
        import subprocess, shlex
        cmd = "python realtime/gateio_kronos_live_improved.py --poll 1 --print-interval 1 --mode rules"
        p = subprocess.run(shlex.split(cmd), timeout=5, capture_output=True, text=True)
        if p.returncode == 0:
            log("子进程结束（可能被超时提前终止），输出预览：")
            print(p.stdout[:800])
            if p.stderr:
                print("[stderr]", p.stderr[:400])
        else:
            log(f"子进程退出码: {p.returncode}")
            print(p.stdout[:400])
            print(p.stderr[:400])
    except Exception as e:
        log(f"试跑失败：{e}")
        traceback.print_exc()

    log("诊断结束。若有失败，请把这段输出贴给我。")

if __name__ == "__main__":
    main()
