
# 可选：你的自定义策略钩子（会被 improved runner 自动调用）
# 输入: candles = [(t, open, high, low, close, volume), ...] 按时间升序
# 输出: dict, 例 {"advice":"做多","reason":"示例: MA金叉","meta":{"ma_fast":..., "ma_slow":...}}

def on_tick(candles):
    if len(candles) < 20:
        return {"advice":"观望", "reason":"样本不足", "meta":{}}
    # 简单均线示例
    closes = [c[4] for c in candles]
    ma_fast = sum(closes[-5:])/5
    ma_slow = sum(closes[-20:])/20
    if ma_fast > ma_slow:
        advice="做多"
        reason="MA5>MA20"
    elif ma_fast < ma_slow:
        advice="做空"
        reason="MA5<MA20"
    else:
        advice="观望"
        reason="均线持平"
    return {"advice":advice, "reason":reason, "meta":{"ma_fast":ma_fast, "ma_slow":ma_slow}}
