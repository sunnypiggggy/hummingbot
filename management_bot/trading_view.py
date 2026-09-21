"""Read-only Grid/DCA cards from a sanitized Report contract."""
import json
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html import escape
from pathlib import Path

from .risk_display import RichText, attention, explanation


def text(value):
    return escape(str(value)[:180]) if value is not None else '未记录'


def when(value):
    try: return datetime.fromtimestamp(float(value),timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S')
    except (TypeError,ValueError,OverflowError): return '未记录'


def amount(value):
    try:
        d=Decimal(str(value))
        if not d.is_finite(): return '未记录'
        return format(d.normalize(),'f')
    except (InvalidOperation,TypeError): return '未记录'


def reference_price(value, *, integer=False):
    try:
        price=Decimal(str(value))
        if not price.is_finite() or price<=0: return '未记录'
        return format(price.quantize(Decimal('1') if integer else Decimal('0.01'), rounding=ROUND_HALF_UP), ',f')
    except (InvalidOperation,TypeError,ValueError): return '未记录'


def progress_lines(executor):
    p=executor.get('progress'); side=executor.get('side')
    lines=[f"\n<b>本轮{'买入' if side=='BUY' else '卖出' if side=='SELL' else '方向未记录'}</b> · <code>{text(str(executor.get('id',''))[-8:])}</code>",
           f"开始：{when(executor.get('started_at'))}"]
    if not isinstance(p,dict): return lines+['本轮实际成交进度尚未接入，不能按挂单数量推算。']
    levels=p.get('levels',[]); adds=levels[1:]
    state={'FILLED':'已完成','PARTIAL':'部分成交','WAITING':'等待成交','UNKNOWN':'待核实'}
    complete=sum(l.get('state')=='FILLED' for l in adds)
    average=reference_price(p.get('average_price'))
    if amount(p.get('open_base'))=='0': average='尚未成交'
    lines += [f"均价：{average} · 已投入：{amount(p.get('open_quote'))} USDT",
              f"首仓：{state.get(levels[0].get('state'),'未记录') if levels else '未记录'} · 补仓完成 {complete}/{len(adds)}档"]
    for level in levels:
        if level.get('state') in ('PARTIAL','UNKNOWN'):
            lines.append(f"第{int(level['index'])+1}档 {state[level['state']]} · 已成交 {amount(level.get('filled_base'))}")
    pending=next((l for l in levels if l.get('state')!='FILLED'),None)
    if pending:
        active=pending.get('active_orders')
        if active:
            lines.append('下一档实际挂单价 '+ ' / '.join(amount(o.get('price')) for o in active))
        elif pending.get('state')=='UNKNOWN':
            lines.append('下一档订单待核实 · 计划价 '+amount(pending.get('price')))
        else:
            lines.append('下一档计划价（未挂出） '+amount(pending.get('price')))
    protection=[]
    references=[]
    for key,label,sign in [('take_profit','止盈',1),('stop_loss','止损',-1)]:
        value=p.get(key)
        if value is None: protection.append(label+'：未设置'); continue
        try:
            ratio=Decimal(value); price=Decimal(p['average_price'])
            direction=1 if side=='BUY' else -1
            reference=amount(price*(1+direction*sign*ratio)) if price>0 and side in ('BUY','SELL') else '未确定'
            protection.append(f'{label} {amount(ratio*100)}%')
            if reference!='未确定': references.append(f'{label}参考价 {reference}')
        except (KeyError,InvalidOperation,TypeError): protection.append(label+'：未记录')
    lines.append('<b>止盈止损</b>：'+' / '.join(protection))
    if references: lines.append(' · '.join(references))
    elif any(p.get(k) is not None for k in ('take_profit','stop_loss')):
        lines.append('触发参考价：成交后确定')
    return lines


def page(root,strategy,index=0,*,reason_cn=lambda s:s,now=None):
    now=time.time() if now is None else now
    route='tv:'+strategy
    rows=[[('刷新',f'{route}:{index}'),('主菜单','m:home')]]
    if strategy not in ('grid','dca'): raise ValueError('unknown_strategy')
    pairs=('BTC-FDUSD','ETH-FDUSD') if strategy=='grid' else ('BTC-USDT','ETH-USDT')
    try:
        doc=json.loads((Path(root)/'management_trading.json').read_text(encoding='utf-8'))
        if doc['schema']!='management-trading-v1' or not 0<=now-float(doc['generated_at'])<=180: raise ValueError('stale')
    except (OSError,ValueError,KeyError,TypeError):
        return RichText(f'<b>{strategy.upper()} · 只读状态</b>\n'+ '\n'.join(p+'：数据不可用（摘要缺失或超过180秒）' for p in pairs)),rows
    header=[f'<b>{strategy.upper()} · 只读状态</b>',f"摘要 {when(doc['generated_at'])}（北京时间）"]
    details=[]
    for pair in pairs:
        row=next((r for r in doc['robots'] if r.get('strategy')==strategy and r.get('pair')==pair),{})
        s=row.get('status') or {}; runtime=row.get('runtime')
        valid=isinstance(runtime,dict) and row.get('snapshot_at') is not None and 0<=now-float(row['snapshot_at'])<=60
        state=('数据不可用' if not valid or not s or s.get('trade_mode')=='UNKNOWN' else
               '停止交易' if s.get('process_running') is False else '正常交易' if s.get('trading_normal') else '交易受限')
        permission=lambda v:'放行' if v is True else '阻止' if v is False else '未知'
        perms=s.get('final_permissions',{}) if valid else {}
        header += [f'<b>{pair}</b>：{state} · 买 {permission(perms.get("buy_enabled"))}／卖 {permission(perms.get("sell_enabled"))}']
        lines=[f'<b>{pair}</b>']
        if not valid: lines.append('交易快照尚未接入或超过60秒，不能确认最新挂单／本轮进度。')
        else:
            lines.append(f"参考价 {reference_price(runtime.get('price'),integer=True)} · 行情 {when(runtime.get('price_observed_at'))}")
            if strategy=='grid':
                orders=runtime.get('orders')
                if not isinstance(orders,list): lines.append('活动订单未记录')
                else:
                    orders=[o for o in orders if o.get('pair')==pair]
                    lines.append(f"买单 {sum(o.get('side')=='BUY' for o in orders)} 笔／卖单 {sum(o.get('side')=='SELL' for o in orders)} 笔")
                    for side in ('BUY','SELL'):
                        selected=sorted((o for o in orders if o.get('side')==side),key=lambda o:Decimal(o['price']),reverse=side=='BUY')
                        for o in selected:
                            lines.append(f"{'买' if side=='BUY' else '卖'} {amount(o.get('price'))} · 剩余 {amount(o.get('remaining'))} · 已成 {amount(o.get('filled'))}"+
                                         (' · 撤单待确认' if o.get('state')=='PENDING_CANCEL' else ''))
            else:
                executors=runtime.get('executors')
                if executors==[]: lines.append('当前没有活动交易轮次')
                elif isinstance(executors,list):
                    for e in executors: lines.extend(progress_lines(e))
                else: lines.append('当前轮次未记录')
        if strategy=='dca':
            h=row.get('history')
            if not h: lines.append('历史成交／轮次数据暂不可用')
            else:
                lines.append('\n<b>成交统计</b>（买 / 卖）')
                for scope,label in [('today','今日'),('total','累计已核实')]:
                    c=h[scope]
                    lines.append(f"{label}：{c['BUY']} / {c['SELL']} 笔")
                lines.append(f"累计起点：{when(h.get('fills_from'))}")
                if h.get('rounds_from') is not None:
                    lines.append('<b>已结束轮次</b>（正常 / 止损 / 其他）')
                    for scope,label in [('today','今日'),('total','累计已核实')]:
                        c=h[scope]; lines.append(f"{label}：{c['normal']} / {c['stop_loss']} / {c['other']}")
                if h.get('unverified_rounds'):
                    lines.append('⚠️ 历史轮次证据不完整，不能作为完整累计。')
                elif h.get('rounds_from') is None: lines.append('已核实结束轮次：0')
                if h.get('residual_rounds'): lines.append(f"⚠️ 尚有残余仓位：{h['residual_rounds']} 轮")
        for gate in s.get('gate_statuses',[]):
            if attention(gate):
                cause,condition,eta=explanation(gate,s,reason_cn)
                lines += ['原因：'+text(cause),'解除条件：'+text(condition),'预计：'+text(eta)]
        details.extend(lines+[''])
    prefix='\n'.join(header)+'\n\n'; pages=[]; current=''
    for line in details:
        if len(prefix)+len(current)+len(line)+40>3500:
            pages.append(current); current=''
        current+=line+'\n'
    pages.append(current); index=max(0,min(int(index),len(pages)-1))
    rows=[[('刷新',f'{route}:{index}'),('主菜单','m:home')]]
    if len(pages)>1: rows.insert(0,[('上一页',f'{route}:{(index-1)%len(pages)}'),('下一页',f'{route}:{(index+1)%len(pages)}')])
    return RichText(prefix+pages[index]+(f'第{index+1}/{len(pages)}页' if len(pages)>1 else '')),rows
