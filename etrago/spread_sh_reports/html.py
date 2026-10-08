"""HTML page assembly: layout, styles and the small chart runtime."""

import html as _html
import json

import numpy as np
import pandas as pd

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def esc(text):
    return _html.escape(str(text))


def fmt(value, digits=2, unit="", signed=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    s = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return f"{s} {unit}" if unit else s


def pct(value, digits=1, signed=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value * 100:+.{digits}f} %" if signed else f"{value * 100:.{digits}f} %"


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def kpi(label, value, sub=None, delta=None, good_when="lower", unit=""):
    """Stat tile. ``delta`` is a number vs. the status quo (same unit)."""
    d = ""
    if delta is not None and np.isfinite(delta) and abs(delta) > 1e-9:
        better = (delta < 0) if good_when == "lower" else (delta > 0) if good_when == "higher" else None
        cls = "neutral" if better is None else ("good" if better else "bad")
        arrow = "▲" if delta > 0 else "▼"
        word = "" if better is None else (" better" if better else " worse")
        d = (f'<div class="delta {cls}"><span aria-hidden="true">{arrow}</span> '
             f'{esc(fmt(delta, 2 if abs(delta) < 10 else 1, unit, signed=True))} vs status quo'
             f'<span class="sr">{word}</span></div>')
    s = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kl">{esc(label)}</div>'
            f'<div class="kv">{esc(value)}</div>{s}{d}</div>')


def kpi_grid(tiles):
    return f'<div class="kpis">{"".join(tiles)}</div>'


def fig_block(fig_id, title, note=None, wide=False, table=True):
    n = f'<p class="note">{note}</p>' if note else ""
    btn = (f'<button class="tbtn" data-table-for="{fig_id}" aria-expanded="false">Data table</button>'
           if table else "")
    return (f'<figure class="card{" wide" if wide else ""}">'
            f'<figcaption><div class="ft">{esc(title)}</div>{btn}</figcaption>{n}'
            f'<div class="chart" id="{fig_id}" role="img" aria-label="{esc(title)}"></div>'
            f'<div class="dtable" id="{fig_id}-table" hidden></div></figure>')


def table_block(df, title=None, note=None, wide=True, formats=None, index=True, max_rows=None):
    formats = formats or {}
    df = df if max_rows is None else df.head(max_rows)
    head = "".join(
        f'<th scope="col">{esc(c)}</th>'
        for c in ([df.index.name or ""] if index else []) + list(df.columns)
    )
    body = []
    for idx, row in df.iterrows():
        cells = [f'<th scope="row">{esc(idx)}</th>'] if index else []
        for c in df.columns:
            v = row[c]
            f = formats.get(c)
            if f is not None and isinstance(v, (int, float, np.floating, np.integer)):
                text = f(v)
                cells.append(f'<td class="num" data-v="{float(v) if np.isfinite(v) else ""}">{esc(text)}</td>')
            elif isinstance(v, (float, np.floating)):
                cells.append(f'<td class="num" data-v="{v if np.isfinite(v) else ""}">{esc(fmt(v))}</td>')
            else:
                cells.append(f"<td>{esc(v)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    t = f'<div class="ft">{esc(title)}</div>' if title else ""
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<div class="card{" wide" if wide else ""}">{t}{n}<div class="tscroll">'
            f'<table class="tbl sortable"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></div>')


def grid(*blocks):
    return f'<div class="grid">{"".join(blocks)}</div>'


def callout(text, kind="info"):
    return f'<div class="callout {kind}">{text}</div>'


def section(sec_id, title, lead, body):
    return (f'<section id="{sec_id}"><h2>{esc(title)}</h2>'
            f'<p class="lead">{lead}</p>{body}</section>')


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
CSS = r"""
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10); --accent:#2a78d6;
  --good:#006300; --bad:#b42828; --callout:#eef4fc; --warnbg:#fdf3e2;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); --accent:#3987e5;
    --good:#0ca30c; --bad:#e66767; --callout:#16212f; --warnbg:#2b2214;
    color-scheme: dark;
  }
}
:root[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); --accent:#3987e5;
  --good:#0ca30c; --bad:#e66767; --callout:#16212f; --warnbg:#2b2214;
  color-scheme: dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}
header.top{position:sticky;top:0;z-index:20;background:var(--surface);
  border-bottom:1px solid var(--ring);padding:10px clamp(16px,3vw,32px);
  display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
header.top .brand{font-weight:650;letter-spacing:.01em}
header.top nav{display:flex;flex-wrap:wrap;gap:4px}
header.top nav a{padding:4px 10px;border-radius:999px;text-decoration:none;color:var(--ink2);
  border:1px solid transparent;font-size:13.5px}
header.top nav a[aria-current="page"]{border-color:var(--ring);color:var(--ink);background:var(--page);font-weight:600}
header.top nav a:hover{color:var(--ink)}
.spacer{flex:1}
button{font:inherit;color:inherit}
.themebtn,.tbtn{background:var(--page);border:1px solid var(--ring);border-radius:8px;padding:4px 10px;
  cursor:pointer;font-size:13px;color:var(--ink2)}
.themebtn:hover,.tbtn:hover{color:var(--ink)}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);gap:28px;
  padding:24px clamp(16px,3vw,32px) 64px;max-width:1480px;margin:0 auto}
body.wide .layout{max-width:1880px;grid-template-columns:190px minmax(0,1fr)}
aside.toc{position:sticky;top:72px;align-self:start;font-size:13.5px}
aside.toc a{display:block;padding:5px 10px;border-left:2px solid var(--grid);color:var(--ink2);text-decoration:none}
aside.toc a:hover,aside.toc a.active{color:var(--ink);border-left-color:var(--accent)}
main{min-width:0}
.hero h1{font-size:clamp(24px,3vw,32px);line-height:1.2;margin:0 0 6px}
.hero p{margin:0 0 4px;color:var(--ink2);max-width:80ch}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 0}
.chip{font-size:12.5px;padding:3px 9px;border-radius:999px;background:var(--surface);border:1px solid var(--ring);color:var(--ink2)}
section{margin-top:44px;scroll-margin-top:72px}
h2{font-size:21px;margin:0 0 6px}
.lead{color:var(--ink2);margin:0 0 16px;max-width:90ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 14px}
.kpi .kl{font-size:12.5px;color:var(--ink2)}
.kpi .kv{font-size:24px;font-weight:650;margin-top:2px;line-height:1.2}
.kpi .sub{font-size:12px;color:var(--muted);margin-top:2px}
.delta{font-size:12px;margin-top:6px;color:var(--ink2)}
.delta.good{color:var(--good)} .delta.bad{color:var(--bad)}
.sr{position:absolute;left:-9999px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:16px 0}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 14px 8px;margin:0;min-width:0}
.card.wide{grid-column:1/-1}
figcaption{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.ft{font-weight:620;font-size:15px}
.note{font-size:13px;color:var(--ink2);margin:4px 0 6px}
.chart{width:100%;min-height:120px}
.dtable{margin:8px 0;max-height:360px;overflow:auto}
.tscroll{overflow-x:auto;margin-top:8px}
table.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
.tbl thead th{font-weight:600;color:var(--ink2);position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none}
.tbl thead th[aria-sort]::after{content:" ↕";color:var(--muted);font-size:11px}
.tbl thead th[aria-sort="ascending"]::after{content:" ↑";color:var(--ink)}
.tbl thead th[aria-sort="descending"]::after{content:" ↓";color:var(--ink)}
.tbl td.num{text-align:right}
.tbl tbody th{font-weight:500}
.callout{background:var(--callout);border-radius:10px;padding:12px 14px;margin:12px 0;font-size:14px;color:var(--ink)}
.callout.warn{background:var(--warnbg)}
.callout ul{margin:6px 0 0 18px;padding:0}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin:16px 0}
.cards a.card{text-decoration:none;color:var(--ink);display:block;padding:14px}
.cards a.card:hover{border-color:var(--accent)}
.cards .cs{font-size:12.5px;color:var(--ink2);margin-top:4px}
footer{color:var(--muted);font-size:12.5px;margin-top:48px}
.seg{display:inline-flex;border:1px solid var(--ring);border-radius:8px;overflow:hidden}
.seg button{background:var(--page);border:0;padding:4px 10px;font-size:13px;color:var(--ink2);cursor:pointer}
.seg button+button{border-left:1px solid var(--ring)}
.seg button[aria-pressed="true"]{background:var(--surface);color:var(--ink);font-weight:600}
.mrow-wrap{overflow-x:auto;margin:10px 0 4px}
.mrow{display:grid;grid-template-columns:repeat(var(--n,5),minmax(190px,1fr));gap:10px}
.mcell{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:8px 8px 6px;min-width:0}
.mcell .mlabel{font-weight:620;font-size:13.5px;margin:0 2px 4px;display:flex;justify-content:space-between;gap:6px}
.mcell .mlabel a{color:var(--ink2);font-weight:400;text-decoration:none;font-size:12px}
.mcell .mkpi{font-size:12.5px;color:var(--ink2);margin:4px 2px 0;line-height:1.35}
.mcell .mkpi b{color:var(--ink);font-size:15px}
.mcell .mkpi .good{color:var(--good)} .mcell .mkpi .bad{color:var(--bad)}
.legendbar{max-width:520px}
.legendbar.chart{min-height:0;height:64px}
.maprow{margin-top:30px;scroll-margin-top:72px}
.maprow h3{font-size:17px;margin:0 0 2px}
.maprow .note{max-width:110ch}
.mini{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin:16px 0}
.mini .card{padding:10px 10px 4px}
.mini .ft{font-size:13.5px}
@media (max-width:980px){
  .layout{grid-template-columns:minmax(0,1fr)}
  aside.toc{display:none}
  .grid{grid-template-columns:minmax(0,1fr)}
}
@media print{header.top,aside.toc,.tbtn,.themebtn{display:none}.layout{display:block}}
"""

RUNTIME = r"""
(function(){
const PAL = {
 light:{s:['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'],
   ink:'#0b0b0b',ink2:'#52514e',muted:'#898781',grid:'#e1e0d9',axis:'#c3c2b7',surface:'#fcfcfb',
   page:'#f9f9f7',land:'#eceae3',landout:'#f4f3ef',border:'#c9c6bb',sea:'#f3f6f9',
   seq:['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'],
   div:['#184f95','#3987e5','#9ec5f4','#ebeae6','#f3aaa0','#e34948','#a32626'],
   divsoft:['#6da7ec','#9ec5f4','#cde2fb','#ebeae6','#f9d3cd','#f3aaa0','#ec8a80']},
 dark:{s:['#3987e5','#d95926','#199e70','#c98500','#d55181','#008300','#9085e9','#e66767'],
   ink:'#ffffff',ink2:'#c3c2b7',muted:'#898781',grid:'#2c2c2a',axis:'#383835',surface:'#1a1a19',
   page:'#0d0d0d',land:'#2a2a28',landout:'#202020',border:'#4a4a46',sea:'#151617',
   seq:['#184f95','#1c5cab','#2a78d6','#3987e5','#6da7ec','#9ec5f4','#cde2fb'],
   div:['#9ec5f4','#3987e5','#1c5cab','#3a3a37','#a33a3a','#e66767','#f6b3b3'],
   divsoft:['#1c5cab','#184f95','#17375f','#3a3a37','#5e2a2a','#7d3030','#a33a3a']}
};
const FIGS = JSON.parse(document.getElementById('fig-data').textContent);
const SHARED = JSON.parse((document.getElementById('shared-data')||{}).textContent||'{}');
const rendered = new Set();
let extentOverride = null;   // {x:[..], y:[..]} applied to maps with spec.extentGroup
let syncing = false;
function theme(){
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark' || t === 'light') return t;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function hex2rgb(h){h=h.replace('#','');return [0,2,4].map(i=>parseInt(h.substr(i,2),16));}
function rgb2hex(c){return '#'+c.map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');}
function ramp(r,t){
  if (!(t>=0)) t=0; if (t>1) t=1;
  const p=t*(r.length-1), i=Math.min(Math.floor(p), r.length-2), f=p-i;
  const a=hex2rgb(r[i]), b=hex2rgb(r[i+1]);
  return rgb2hex(a.map((v,k)=>v+(b[k]-v)*f));
}
function scale(r){return r.map((c,i)=>[i/(r.length-1),c]);}
function tok(s,T){
  if (typeof s!=='string' || s[0]!=='@') return s;
  const parts=s.slice(1).split(':'), k=parts[0], v=parts.slice(1).join(':');
  if (k[0]==='s' && /^s\d$/.test(k)) return T.s[parseInt(k[1])-1];
  if (k==='seq') return ramp(T.seq, parseFloat(v));
  if (k==='div') return ramp(T.div, parseFloat(v));
  if (k==='seqscale') return scale(T.seq);
  if (k==='divscale') return scale(T.div);
  if (k==='divsoftscale') return scale(T.divsoft);
  if (k==='fade') { const c=hex2rgb(tok('@'+v,T)); return `rgba(${c[0]},${c[1]},${c[2]},0.25)`; }
  return T[k]!==undefined ? T[k] : s;
}
function resolve(o,T){
  if (Array.isArray(o)) return o.map(x=>resolve(x,T));
  if (o && typeof o==='object'){const r={};for(const k in o) r[k]=resolve(o[k],T);return r;}
  return tok(o,T);
}
function merge(a,b){
  const r=Object.assign({},a);
  for (const k in b){
    if (b[k] && typeof b[k]==='object' && !Array.isArray(b[k]) && a[k] && typeof a[k]==='object') r[k]=merge(a[k],b[k]);
    else r[k]=b[k];
  }
  return r;
}
function base(T,kind){
  const axis={gridcolor:T.grid,linecolor:T.axis,zerolinecolor:T.axis,tickcolor:T.axis,
    tickfont:{color:T.ink2,size:11.5},title:{font:{color:T.ink2,size:12}},automargin:true};
  return {
    paper_bgcolor:T.surface, plot_bgcolor: kind==='map'?T.sea:T.surface,
    font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif',color:T.ink2,size:12.5},
    margin: kind==='map'?{l:4,r:4,t:4,b:4}:{l:10,r:14,t:10,b:10},
    xaxis:axis, yaxis:axis, bargap:0.28, bargroupgap:0.08, barcornerradius:4,
    legend:{orientation:'h',y:1.02,yanchor:'bottom',x:0,font:{color:T.ink2,size:12},bgcolor:'rgba(0,0,0,0)'},
    hoverlabel:{bgcolor:T.surface,bordercolor:T.axis,font:{color:T.ink,size:12.5}},
    colorway:T.s
  };
}
function draw(id){
  const el=document.getElementById(id), spec=FIGS[id];
  if (!el || !spec || typeof Plotly==='undefined') return;
  const T=PAL[theme()];
  const raw = spec.bg && SHARED[spec.bg] ? SHARED[spec.bg].concat(spec.data) : spec.data;
  const data=resolve(raw,T);
  data.forEach(tr=>{
    if (tr.type==='bar'){ tr.marker=tr.marker||{}; if(!tr.marker.line) tr.marker.line={color:T.surface,width:1.5}; }
    if (tr.type==='scatter' && (tr.mode||'').includes('lines') && spec.kind!=='map' && tr.line && !tr.line.width) tr.line.width=2;
  });
  const layout=merge(base(T,spec.kind), resolve(spec.layout,T));
  layout.height=spec.height; layout.autosize=true;
  if (spec.kind==='map' && spec.extentGroup && extentOverride && extentOverride[spec.extentGroup]){
    const e=extentOverride[spec.extentGroup];
    layout.xaxis.range=e.x.slice(); layout.yaxis.range=e.y.slice();
    layout.yaxis.scaleratio=e.ratio;
  }
  Plotly.react(el,data,layout,{responsive:true,displaylogo:false,
    displayModeBar: spec.compact ? false : 'hover',
    modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d'],
    scrollZoom: spec.kind==='map', toImageButtonOptions:{format:'png',scale:2,filename:id}});
  if (spec.group && !el.dataset.linked){
    el.dataset.linked='1';
    // Linked pan/zoom: every map of the same row follows.
    el.on('plotly_relayout', ev=>{
      if (syncing) return;
      const x0=ev['xaxis.range[0]'], x1=ev['xaxis.range[1]'], y0=ev['yaxis.range[0]'], y1=ev['yaxis.range[1]'];
      const upd = (x0!==undefined && y0!==undefined) ? {'xaxis.range':[x0,x1],'yaxis.range':[y0,y1]}
                : ev['xaxis.autorange'] ? {'xaxis.range':spec.layout.xaxis.range,'yaxis.range':spec.layout.yaxis.range} : null;
      if (!upd) return;
      syncing=true;
      const jobs=[];
      Object.keys(FIGS).forEach(o=>{ if(o!==id && FIGS[o].group===spec.group && rendered.has(o)) jobs.push(Plotly.relayout(o,upd)); });
      Promise.all(jobs).finally(()=>{ syncing=false; });
    });
  }
  rendered.add(id);
}
window.SSHR = {
  setExtent(group, x, y){
    const lat=(y[0]+y[1])/2, ratio=1/Math.cos(lat*Math.PI/180);
    extentOverride = extentOverride || {};
    extentOverride[group] = {x:x, y:y, ratio:ratio};
    syncing=true;
    const jobs=[];
    rendered.forEach(id=>{ if(FIGS[id].extentGroup===group) jobs.push(Plotly.relayout(id,{'xaxis.range':x.slice(),'yaxis.range':y.slice(),'yaxis.scaleratio':ratio})); });
    Promise.all(jobs).finally(()=>{ syncing=false; });
  }
};
// ?extent=sh (or the button index) preselects a map extent
(function(){
  const m=location.search.match(/[?&]extent=(\w+)/); if(!m) return;
  const btns=[...document.querySelectorAll('[data-extent]')]; if(!btns.length) return;
  const b = m[1]==='sh' ? btns.find(x=>/Schleswig/.test(x.textContent)) : btns[parseInt(m[1])||0];
  if(!b) return;
  const v=JSON.parse(b.dataset.extent), lat=(v.y[0]+v.y[1])/2;
  extentOverride={}; extentOverride[v.group]={x:v.x,y:v.y,ratio:1/Math.cos(lat*Math.PI/180)};
  btns.forEach(o=>o.setAttribute('aria-pressed', String(o===b)));
})();
document.querySelectorAll('[data-extent]').forEach(b=>b.addEventListener('click',()=>{
  const v=JSON.parse(b.dataset.extent);
  document.querySelectorAll('[data-extent]').forEach(o=>o.setAttribute('aria-pressed', String(o===b)));
  window.SSHR.setExtent(v.group, v.x, v.y);
}));
function redrawAll(){ rendered.forEach(draw); }
const io = ('IntersectionObserver' in window) ? new IntersectionObserver(es=>{
  es.forEach(e=>{ if(e.isIntersecting){ io.unobserve(e.target); draw(e.target.id);} });
},{rootMargin:'400px'}) : null;
function start(){
  // ?all renders every chart immediately (printing, screenshots)
  const eager = /[?&]all\b/.test(location.search) || matchMedia('print').matches;
  document.querySelectorAll('.chart').forEach(el=>{ if(io && !eager) io.observe(el); else draw(el.id); });
  window.addEventListener('beforeprint', ()=>document.querySelectorAll('.chart').forEach(el=>{ if(!rendered.has(el.id)) draw(el.id); }));
}
// Theme toggle: system -> light -> dark
const btn=document.getElementById('theme');
function label(){ const t=document.documentElement.getAttribute('data-theme'); btn.textContent='Theme: '+(t||'system'); }
if (btn){ label(); btn.addEventListener('click',()=>{
  const r=document.documentElement, t=r.getAttribute('data-theme');
  const next = !t ? 'light' : t==='light' ? 'dark' : null;
  if (next) r.setAttribute('data-theme',next); else r.removeAttribute('data-theme');
  try{ localStorage.setItem('sshr-theme', next||''); }catch(e){}
  label(); redrawAll();
});}
try{ const saved=localStorage.getItem('sshr-theme'); if(saved){document.documentElement.setAttribute('data-theme',saved); if(btn) label();} }catch(e){}
// ?theme=dark|light forces a theme (sharing links, screenshots)
const forced=(location.search.match(/[?&]theme=(dark|light)\b/)||[])[1];
if (forced){ document.documentElement.setAttribute('data-theme',forced); if(btn) label(); }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
new MutationObserver(redrawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
// Data tables built from the figure traces
function fmtv(v){ return (typeof v==='number') ? v.toLocaleString(undefined,{maximumFractionDigits:3}) : (v==null?'–':String(v)); }
document.querySelectorAll('[data-table-for]').forEach(b=>b.addEventListener('click',()=>{
  const id=b.dataset.tableFor, box=document.getElementById(id+'-table'), spec=FIGS[id];
  const open=box.hidden; box.hidden=!open; b.setAttribute('aria-expanded',String(open));
  if (!open || box.dataset.built) return;
  const tr=spec.data.filter(t=>t.name && (t.type==='bar'||t.type==='scatter'||t.type==='histogram') && (t.x||t.y));
  if (spec.data[0] && spec.data[0].type==='heatmap'){
    const h=spec.data[0]; let s='<table class="tbl"><thead><tr><th></th>'+h.x.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
    h.y.forEach((y,i)=>{ s+='<tr><th>'+y+'</th>'+h.z[i].map(v=>'<td class="num">'+fmtv(v)+'</td>').join('')+'</tr>'; });
    box.innerHTML=s+'</tbody></table>'; box.dataset.built=1; return;
  }
  if (!tr.length){ box.innerHTML='<p class="note">See the tables in this section for the underlying values.</p>'; box.dataset.built=1; return; }
  const horiz=tr[0].orientation==='h';
  const cats=(horiz?tr[0].y:tr[0].x)||[];
  if (cats.length>400){ box.innerHTML='<p class="note">Too many points for a table ('+cats.length+'); hover the chart for values.</p>'; box.dataset.built=1; return; }
  let s='<table class="tbl"><thead><tr><th></th>'+tr.map(t=>'<th>'+t.name+'</th>').join('')+'</tr></thead><tbody>';
  cats.forEach((c,i)=>{ s+='<tr><th>'+c+'</th>'+tr.map(t=>'<td class="num">'+fmtv((horiz?t.x:t.y)[i])+'</td>').join('')+'</tr>'; });
  box.innerHTML=s+'</tbody></table>'; box.dataset.built=1;
}));
// Sortable tables
document.querySelectorAll('table.sortable').forEach(t=>{
  t.querySelectorAll('thead th').forEach((th,ci)=>{
    th.setAttribute('aria-sort','none');
    th.addEventListener('click',()=>{
      const dir = th.getAttribute('aria-sort')==='descending' ? 'ascending' : 'descending';
      t.querySelectorAll('thead th').forEach(o=>o.setAttribute('aria-sort','none'));
      th.setAttribute('aria-sort',dir);
      const rows=[...t.tBodies[0].rows];
      rows.sort((a,b)=>{
        const A=a.cells[ci], B=b.cells[ci];
        const va=A.dataset.v!==undefined?parseFloat(A.dataset.v):A.textContent, vb=B.dataset.v!==undefined?parseFloat(B.dataset.v):B.textContent;
        const r=(typeof va==='number'&&typeof vb==='number') ? ((isNaN(va)?-Infinity:va)-(isNaN(vb)?-Infinity:vb)) : String(va).localeCompare(String(vb));
        return dir==='ascending'?r:-r;
      });
      rows.forEach(r=>t.tBodies[0].appendChild(r));
    });
  });
});
// Active section in the table of contents
const links=[...document.querySelectorAll('aside.toc a')];
if ('IntersectionObserver' in window && links.length){
  const so=new IntersectionObserver(es=>es.forEach(e=>{ if(e.isIntersecting){
    links.forEach(a=>a.classList.toggle('active', a.getAttribute('href')==='#'+e.target.id)); }}),{rootMargin:'-30% 0px -60% 0px'});
  document.querySelectorAll('main section').forEach(s=>so.observe(s));
}
if (typeof Plotly==='undefined'){
  document.querySelectorAll('.chart').forEach(el=>{ el.innerHTML='<p class="note">Charts need plotly.js. Connect to the internet (CDN) or rebuild with --offline-js.</p>'; });
} else start();
})();
"""


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def page(title, heading, intro, chips, nav, toc, sections_html, figs, plotly_js=None,
         footer="", shared=None, controls="", body_class=""):
    fig_json = json.dumps(_clean(figs), default=_json_default, separators=(",", ":"))
    fig_json = fig_json.replace("</", "<\\/")
    shared_json = json.dumps(_clean(shared or {}), default=_json_default, separators=(",", ":"))
    shared_json = shared_json.replace("</", "<\\/")
    current = ' aria-current="page"'
    nav_html = "".join(
        f'<a href="{esc(href)}"{current if cur else ""}>{esc(text)}</a>'
        for text, href, cur in nav
    )
    toc_html = "".join(f'<a href="#{sid}">{esc(t)}</a>' for sid, t in toc)
    chips_html = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
    script = (f"<script>{plotly_js}</script>" if plotly_js
              else f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body class="{body_class}">
<header class="top">
  <span class="brand">SPREAD.SH · bidding zones</span>
  <nav aria-label="Reports">{nav_html}</nav>
  <span class="spacer"></span>
  {controls}
  <button class="themebtn" id="theme" type="button">Theme: system</button>
</header>
<div class="layout">
  <aside class="toc" aria-label="Contents">{toc_html}</aside>
  <main>
    <div class="hero"><h1>{esc(heading)}</h1><p>{intro}</p><div class="meta">{chips_html}</div></div>
    {sections_html}
    <footer>{footer}</footer>
  </main>
</div>
{script}
<script type="application/json" id="fig-data">{fig_json}</script>
<script type="application/json" id="shared-data">{shared_json}</script>
<script>{RUNTIME}</script>
</body>
</html>
"""
