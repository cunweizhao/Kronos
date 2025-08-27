# Windows PowerShell 一键启动
Set-Location -Path $PSScriptRoot
python .\RUN_ALL_IN_ONE.py --symbol ETH_USDT --interval 10s --poll 3 --print-interval 10 --mode both
