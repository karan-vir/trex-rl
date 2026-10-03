"""Live training dashboard.   python scripts/monitor.py   ->   open http://127.0.0.1:8777

Reads runs/*/eval.csv (written by scripts/train_ppo.py) every few seconds and shows the evaluation
scores over training, whether a training process is still running, progress and an ETA.
Read-only: it never touches the training processes.
"""

import argparse
import csv
import http.server
import json
import os
import re
import socketserver
import subprocess
import time
from pathlib import Path

RUNS = Path("runs")


def running_jobs():
    """{run name: total frames} for every live train_ppo.py process."""
    out = {}
    try:
        ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return out
    for line in ps.splitlines():
        if "train_ppo.py" in line and "--name" in line and "multiprocessing" not in line:
            name = re.search(r"--name\s+(\S+)", line)
            frames = re.search(r"--frames\s+(\d+)", line)
            if name:
                out[name.group(1)] = int(frames.group(1)) if frames else None
    return out


def collect():
    jobs = running_jobs()
    runs = []
    for d in sorted(RUNS.glob("*/eval.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        rows = []
        with open(d) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({k: float(v) for k, v in r.items() if k != "frac_cap"})
                except ValueError:
                    pass
        name = d.parent.name
        runs.append({"name": name, "rows": rows, "running": name in jobs, "target": jobs.get(name),
                     "age_s": round(time.time() - d.stat().st_mtime)})
    return {"runs": runs, "now": time.strftime("%H:%M:%S")}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training monitor</title>
<style>
:root{--bg:#fafaf8;--card:#fff;--ink:#1c1c1a;--mute:#6b6b66;--line:#e4e3de;--c1:#2563eb;--c2:#0f9d58;--c3:#d97706;--c4:#9333ea;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1d1d1b;--ink:#ecebe6;--mute:#9a9992;--line:#34332f;--c1:#6ea0ff;--c2:#4cc38a;--c3:#f0a54a;--c4:#c08cf5;--bad:#f06a6a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:12px}
.row{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat{flex:1 1 130px}.stat .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.stat .l{color:var(--mute);font-size:12px}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid var(--line)}
.run{background:rgba(15,157,88,.14);color:var(--c2)}.done{color:var(--mute)}.stall{background:rgba(220,38,38,.12);color:var(--bad)}
.bar{height:8px;background:var(--line);border-radius:8px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--c1)}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit}
canvas{width:100%;height:300px;display:block}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--mute);font-weight:500;font-size:12px}
.legend span{margin-right:14px;font-size:12px;color:var(--mute)}.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
.note{color:var(--mute);font-size:12px;margin-top:8px}
</style></head><body><div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
 <div><h1>Training monitor</h1><div class="sub" id="upd">loading…</div></div>
 <div><label class="sub">run </label><select id="sel"></select> <span class="chip" id="chip"></span></div></div>
<div class="row card" style="flex-direction:column;gap:6px"><div style="display:flex;justify-content:space-between"><span id="prog">–</span><span class="sub" id="eta"></span></div><div class="bar"><i id="barfill" style="width:0"></i></div></div>
<div class="row" id="stats"></div>
<div class="card"><div class="legend" id="legend"></div><canvas id="c"></canvas>
 <div class="note">Each point = 24 fixed-seed games, 120 Hz timing, no randomisation, deterministic play. A score of ≈6,195 means the game reached the 20,000-decision cap (survived), so the median saturates there; watch the <b>10th percentile</b> (how bad the unlucky games are) and the <b>mean</b>.</div></div>
<div class="card" style="margin-top:12px;overflow-x:auto"><table id="tbl"></table></div>
</div>
<script>
const $=id=>document.getElementById(id);let DATA=null,sel=null;
const fmt=n=>Math.round(n).toLocaleString();
async function load(){try{DATA=await (await fetch('/data',{cache:'no-store'})).json();render()}catch(e){$('upd').textContent='cannot reach the monitor server'}}
function render(){
 const runs=DATA.runs;if(!runs.length){$('upd').textContent='no runs yet';return}
 const s=$('sel');if(!s.options.length||s.options.length!==runs.length){s.innerHTML=runs.map(r=>`<option>${r.name}</option>`).join('');s.onchange=()=>{sel=s.value;render()}}
 if(!sel||!runs.find(r=>r.name===sel))sel=(runs.find(r=>r.running)||runs[0]).name;s.value=sel;
 const r=runs.find(x=>x.name===sel),rows=r.rows,last=rows[rows.length-1];
 $('upd').textContent='updated '+DATA.now+' · refreshes every 5 s';
 const stalled=r.running&&r.age_s>600;
 $('chip').className='chip '+(stalled?'stall':r.running?'run':'done');$('chip').textContent=stalled?'stalled?':r.running?'training':'finished';
 if(last){const done=last.frames,tot=r.target;const pct=tot?Math.min(100,done/tot*100):0;
  $('prog').innerHTML=`<b>${(done/1e6).toFixed(1)}M</b> ${tot?'of '+(tot/1e6).toFixed(0)+'M frames':'frames'} · ${last.minutes.toFixed(1)} min`;
  $('barfill').style.width=pct+'%';
  if(r.running&&tot&&done>0){const rate=done/last.minutes;const left=(tot-done)/rate;$('eta').textContent='about '+(left<1?'<1':Math.round(left))+' min left'}else $('eta').textContent=r.running?'':'';
  const best=Math.max(...rows.map(x=>x.mean));
  $('stats').innerHTML=[['median',last.median],['mean',last.mean],['10th percentile',last.p10],['90th percentile',last.p90],['best mean so far',best]].map(([l,v])=>`<div class="card stat"><div class="v">${fmt(v)}</div><div class="l">${l}</div></div>`).join('')}
 $('legend').innerHTML=[['median','--c1'],['mean','--c2'],['10th percentile (worst games)','--c3'],['90th percentile','--c4']].map(([l,c])=>`<span><i style="background:var(${c})"></i>${l}</span>`).join('');
 draw(rows);
 $('tbl').innerHTML='<tr><th>frames (M)</th><th>minutes</th><th>median</th><th>mean</th><th>p10</th><th>p90</th><th>best game</th></tr>'+rows.slice(-12).reverse().map(x=>`<tr><td>${(x.frames/1e6).toFixed(1)}</td><td>${x.minutes.toFixed(1)}</td><td>${fmt(x.median)}</td><td>${fmt(x.mean)}</td><td>${fmt(x.p10)}</td><td>${fmt(x.p90)}</td><td>${fmt(x.max)}</td></tr>`).join('');
}
function draw(rows){const cv=$('c'),dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);
 const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
 if(rows.length<1){return}
 const L=52,R=10,T=10,B=26,xs=rows.map(r=>r.frames),xmax=Math.max(...xs)*1.0001,ymax=Math.max(6500,...rows.map(r=>r.p90))*1.05;
 const X=v=>L+(v/xmax)*(W-L-R),Y=v=>T+(1-v/ymax)*(H-T-B);
 g.strokeStyle=css('--line');g.fillStyle=css('--mute');g.font='11px system-ui';g.lineWidth=1;
 for(let i=0;i<=4;i++){const v=ymax*i/4,y=Y(v);g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();g.textAlign='right';g.fillText(fmt(v),L-6,y+4)}
 g.textAlign='center';const n=Math.min(6,rows.length);for(let i=0;i<=n;i++){const v=xmax*i/n;g.fillText((v/1e6).toFixed(0)+'M',X(v),H-8)}
 const series=[['p90','--c4',1,[]],['p10','--c3',2,[5,4]],['mean','--c2',2,[]],['median','--c1',2.5,[]]];
 for(const [k,c,w,dash] of series){g.strokeStyle=css(c);g.lineWidth=w;g.setLineDash(dash);g.beginPath();rows.forEach((r,i)=>{const x=X(r.frames),y=Y(r[k]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
 g.setLineDash([]);const last=rows[rows.length-1];g.fillStyle=css('--c1');g.beginPath();g.arc(X(last.frames),Y(last.median),4,0,7);g.fill();
}
window.addEventListener('resize',()=>DATA&&render());load();setInterval(load,5000);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data"):
            body, ctype = json.dumps(collect()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", a.port), Handler) as srv:
        print(f"training monitor: http://127.0.0.1:{a.port}", flush=True)
        srv.serve_forever()
