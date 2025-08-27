#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行脚本（Windows / macOS / Linux 通用）
================================================
功能：
1) 自动创建虚拟环境 .venv
2) 自动安装依赖（含 torch，尽量安装 CPU 版）
3) 运行改进后的 Gate.io 实盘监控脚本：
   Kronos/Kronos/realtime/gateio_kronos_live_improved.py
   - 默认：ETH_USDT, 10s K线, 每3秒轮询，每10秒打印（北京时间）
   - 若网络/SSL受限，会自动降级 MOCK 继续跑，流程不断

用法：
  在项目根（与 Kronos/ 同级）执行：
    python RUN_ALL_IN_ONE.py
可选参数：
    --symbol ETH_USDT --interval 10s --poll 3 --print-interval 10 --mode both
"""

import os, sys, subprocess, shlex, venv, platform, ssl, time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.join(THIS_DIR, "Kronos", "Kronos")
RUNNER = os.path.join(KRONOS_DIR, "realtime", "gateio_kronos_live_improved.py")
VENV_DIR = os.path.join(THIS_DIR, ".venv")

def run(cmd, env=None, check=True):
    print(f"[RUN] {cmd}")
    r = subprocess.run(shlex.split(cmd), env=env, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"命令失败（退出码 {r.returncode}）：{cmd}")
    return r.returncode

def ensure_venv():
    if not os.path.isdir(VENV_DIR):
        print(f"创建虚拟环境：{VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"虚拟环境已存在：{VENV_DIR}")

def python_bin():
    if platform.system().lower().startswith("win"):
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def pip_bin():
    if platform.system().lower().startswith("win"):
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")

def install_requirements():
    py = python_bin()
    pip = pip_bin()
    run(f"{py} -m pip install --upgrade pip setuptools wheel")
    # 尝试安装 CPU 版 torch（最佳努力）
    try:
        run(f'{py} -m pip install --index-url https://download.pytorch.org/whl/cpu torch', check=False)
    except Exception:
        pass
    # 其他依赖（尽量用常规源）
    base_reqs = [
        "pandas",
        "numpy",
        "requests",
        "transformers",
        "huggingface_hub",
        # 可选：可视化/日志，如需： "matplotlib", "tqdm"
    ]
    run(f"{py} -m pip install " + " ".join(base_reqs), check=False)

def main():
    # 参数解析（透传给 runner）
    args = sys.argv[1:]
    # 1) 确认 runner 存在
    if not os.path.exists(RUNNER):
        raise SystemExit(f"未找到运行脚本：{RUNNER}，请确认当前目录包含 Kronos/Kronos")
    # 2) venv
    ensure_venv()
    # 3) 依赖
    install_requirements()
    # 4) 运行
    py = python_bin()
    env = os.environ.copy()
    # 可选：关闭 HF telemetry & 设置缓存目录，避免权限与公司代理影响
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HOME", os.path.join(THIS_DIR, ".hf_cache"))
    # 某些公司 CA 证书会导致 SSL 报错，可临时放宽（仅排错用；若需要，把下行改为 "0"）
    env.setdefault("PYTHONHTTPSVERIFY", "1")
    cmd = f'{py} "{RUNNER}" --symbol ETH_USDT --interval 10s --poll 3 --print-interval 10 --mode both'
    # 若传入自定义参数，覆盖默认命令
    if args:
        # 将默认命令替换为用户参数
        cmd = f'{py} "{RUNNER}" ' + " ".join(args)
    print("\n=== 开始运行 Gate.io 实盘监控（若首次，会先下载/安装依赖）===\n")
    run(cmd, env=env, check=False)
    print("\n=== 运行结束（如果中途报错，请把报错粘给我）===\n")

if __name__ == "__main__":
    main()
