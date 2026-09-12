"""Mobile 168-hour observed probability/step-threshold chart; never bridge gaps."""
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def snapshot(contract, strategy, asset, now):
    if strategy not in {"grid", "dca"} or asset not in {"BTC", "ETH"}:
        raise ValueError("不支持的机器人")
    if contract.get("schema") != "management-probability-history-v1":
        raise ValueError("历史曲线合同不可用")
    source = contract.get("pairs", {}).get(f"{asset}-FDUSD", {})
    points = []
    for row in source.get("points", []):
        try:
            if not all(not isinstance(row[k], bool) and math.isfinite(float(row[k]))
                       for k in ("signal_ts", "probability", "threshold")):
                continue
            if (now-168*3600 <= row["signal_ts"] <= now and
                    0 <= row["probability"] <= 1 and 0 <= row["threshold"] <= 1):
                points.append(dict(row))
        except (KeyError, TypeError, ValueError):
            continue
    points.sort(key=lambda r: r["signal_ts"])
    if not points:
        raise ValueError("过去一周没有可信线上信号记录，正在积累历史；未发送旧图")
    for a, b in zip(points, points[1:]):
        if b["signal_ts"] <= a["signal_ts"]:
            raise ValueError("历史信号时间重复，无法绘图")
    fresh = 0 <= now-float(contract.get("generated_at", 0)) <= 300
    market_pair = f"{asset}-{'FDUSD' if strategy == 'grid' else 'USDT'}"
    prices = {}
    for row in contract.get('market_prices', {}).get(market_pair, {}).get('points', []):
        stamp, close = row.get('timestamp'), row.get('close')
        if (all(type(v) in (int, float) and math.isfinite(v) for v in (stamp,close))
                and now-168*3600 <= stamp <= now and close > 0):
            prices[stamp] = {'timestamp': stamp, 'close': close}
    return dict(strategy=strategy, pair=f"{asset}-{'FDUSD' if strategy == 'grid' else 'USDT'}",
                prices=[prices[t] for t in sorted(prices)],
                source=f"{asset}-FDUSD", points=points, start=now-168*3600, end=now,
                available=fresh and source.get("current_available") is True,
                recovery=source.get("current_recovery") if fresh and source.get("current_available") is True else None)


def recovery_label(data):
    recovery = data.get('recovery')
    if not isinstance(recovery, dict):
        return '当前恢复计数：暂无可信数据'
    count, required = recovery.get('count'), recovery.get('ordinary_required')
    if type(count) is not int or count < 0 or type(required) is not int or required <= 0:
        return '当前恢复计数：暂无可信数据'
    if recovery.get('risk_off') is False:
        return '当前恢复计数：不适用（模型已放行）'
    if recovery.get('risk_off') is not True:
        return '当前恢复计数：暂无可信数据'
    return f'当前普通恢复计数：{count} / {required} 次（完整4小时周期）'


def segments(data):
    result = []
    for row in data["points"]:
        if (not result or row.get("break_before") or
                row["signal_ts"]-result[-1][-1]["signal_ts"] > 3600):
            result.append([])
        result[-1].append(row)
    return result


def render(data, path):
    font_path = next((p for p in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc")) if p.exists()), None)
    if font_path is None:
        raise ValueError("中文绘图字体未安装")
    im = Image.new("RGB", (1440, 1800), "white")
    d = ImageDraw.Draw(im)
    def text(x, y, s, size=34, fill="#222222", anchor=None):
        d.text((x,y), s, font=ImageFont.truetype(str(font_path),size), fill=fill, anchor=anchor)
    def bj(t):
        return datetime.fromtimestamp(t, timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
    rows = data["points"]
    text(70,45,f"{data['strategy'].upper()} {data['pair']}",52)
    text(70,125,"近一周价格、v22 概率与阈值",48)
    text(70,205,f"请求：{bj(data['start'])} — {bj(data['end'])}（北京时间）")
    text(70,270,f"记录覆盖：{bj(rows[0]['signal_ts'])} — {bj(rows[-1]['signal_ts'])}")
    text(70,335,f"最新记录：概率 {rows[-1]['probability']:.2%} / 阈值 {rows[-1]['threshold']:.2%}")
    text(70,400,"当前信号：" + ("可用" if data["available"] else "不可用；下图仅为已保存历史"))
    text(70,450,recovery_label(data),32)
    left,right,top,bottom=175,1360,1120,1660
    x=lambda t: left+(t-data["start"])/(168*3600)*(right-left)
    y=lambda v: bottom-v*(bottom-top)
    prices = data.get('prices', [])
    text(70,525,f"价格 · {data['pair']} · 1小时收盘价",34)
    if prices:
        text(70,575,f"最近收盘 {prices[-1]['close']:,.2f}  ·  {bj(prices[-1]['timestamp'])}",30)
        low, high = min(r['close'] for r in prices), max(r['close'] for r in prices)
        pad = max((high-low)*.08, high*.0001)
        low, high = low-pad, high+pad
        py = lambda v: 950-(v-low)/(high-low)*300
        for i in range(4):
            value = low+(high-low)*i/3
            d.line((left,py(value),right,py(value)),fill='#DFE2E5',width=2)
            text(left-15,py(value),f'{value:,.0f}',26,anchor='rm')
        for a,b in zip(prices,prices[1:]):
            if b['timestamp']-a['timestamp'] <= 3600:
                d.line((x(a['timestamp']),py(a['close']),x(b['timestamp']),py(b['close'])),fill='#AE7329',width=5)
        for row in prices:
            px, yy = x(row['timestamp']), py(row['close'])
            d.ellipse((px-3,yy-3,px+3,yy+3),fill='#AE7329')
    else:
        text(200,780,'暂无可信历史行情',36)
    d.line((left,650,left,950,right,950),fill='#535960',width=2)
    for i in range(8):
        text(x(data['start']+i*86400),975,bj(data['start']+i*86400)[:5],26,anchor='mt')
    def dash(a,b):
        if a[1] == b[1]:
            for px in range(int(a[0]), int(b[0])+1):
                if px % 20 < 11:
                    d.line((px,a[1],px,a[1]+3),fill="#535960",width=1)
            return
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if not length:
            return
        for start in range(0,int(length)+1,18):
            end=min(start+10,length)
            d.line((a[0]+(b[0]-a[0])*start/length,a[1]+(b[1]-a[1])*start/length,
                    a[0]+(b[0]-a[0])*end/length,a[1]+(b[1]-a[1])*end/length),
                   fill="#535960",width=4)
    for group in segments(data):
        for a,b in zip(group,group[1:]):
            if a.get("risk_off") == 1 and b.get("risk_off") == 1:
                d.rectangle((x(a["signal_ts"]),top,x(b["signal_ts"]),bottom),fill="#F8ECDD")
    for v in (0,.25,.5,.75,1):
        d.line((left,y(v),right,y(v)),fill="#DFE2E5",width=2)
        text(left-20,y(v),f"{v:.0%}",28,anchor="rm")
    for i in range(8):
        t=data["start"]+i*86400
        text(x(t),bottom+30,bj(t)[:5],26,anchor="mt")
    d.line((left,top,left,bottom,right,bottom),fill="#535960",width=2)
    for group in segments(data):
        for a,b in zip(group,group[1:]):
            d.line((x(a["signal_ts"]),y(a["probability"]),x(b["signal_ts"]),y(b["probability"])),
                   fill="#3568AC",width=5)
            dash((x(a["signal_ts"]),y(a["threshold"])),(x(b["signal_ts"]),y(a["threshold"])))
            dash((x(b["signal_ts"]),y(a["threshold"])),(x(b["signal_ts"]),y(b["threshold"])))
        for r in group:
            px,py=x(r["signal_ts"]),y(r["probability"])
            d.ellipse((px-4,py-4,px+4,py+4),fill="#3568AC")
            tx,ty=px,y(r["threshold"])
            d.rectangle((tx-3,ty-3,tx+3,ty+3),fill="#535960")
    for a,b in zip(rows,rows[1:]):
        if a.get("week") != b.get("week"):
            px=x(b["signal_ts"])
            d.line((px,top,px,bottom),fill="#AAAAAA",width=2)
            text(min(right-130,max(left,px)),top-45,f"第{b['week']}周",26)
    text(70,1030,"概率（蓝） / 阈值（灰虚线） / Risk-Off（浅色阴影）",30)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    im.save(path,format="PNG")
    return path
