"""A tiny public browser demo for the orchestrator.

Serves a single HTML page with a form. The page calls this service's own
`/api/verify`, which proxies to the orchestrator's bearer-protected `/verify`
with the token held **server-side** (env API_AUTH_TOKEN) — so the token is never
exposed to the browser. Deployed as its own Cloud Run service (ROLE=demo).
"""

from __future__ import annotations

import json
import os

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080").rstrip("/")
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "")

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Truthfulness Verifier</title>
<style>
 :root{--bg:#0f172a;--card:#1e293b;--mut:#94a3b8;--line:#334155;--accent:#6366f1}
 *{box-sizing:border-box} body{margin:0;font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;
   background:var(--bg);color:#e2e8f0;display:flex;justify-content:center;padding:32px}
 .wrap{width:100%;max-width:720px}
 h1{font-size:24px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 24px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}
 label{display:block;font-size:13px;color:var(--mut);margin:12px 0 4px}
 textarea,input{width:100%;background:#0b1220;border:1px solid var(--line);color:#e2e8f0;
   border-radius:8px;padding:10px;font:inherit} textarea{min-height:80px;resize:vertical}
 .row{display:flex;gap:12px} .row>div{flex:1}
 button{margin-top:16px;background:var(--accent);color:#fff;border:0;border-radius:8px;
   padding:11px 18px;font-weight:600;cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 .ex{font-size:13px;color:var(--mut);margin-top:10px} .ex a{color:#a5b4fc;cursor:pointer;text-decoration:underline}
 .badge{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;font-size:14px}
 .t{background:#064e3b;color:#6ee7b7} .f{background:#7f1d1d;color:#fca5a5}
 .meta{color:var(--mut);font-size:13px;margin:8px 0} .expl{margin-top:8px}
 .hide{display:none} .err{color:#fca5a5}
</style></head><body><div class="wrap">
 <h1>Truthfulness Verifier</h1>
 <p class="sub">A multi-agent service (A2A + MCP) on Cloud Run. Enter a public statement; the
   orchestrator asks a zero-shot and a fine-tuned agent, reconciles them, and explains the verdict.</p>
 <div class="card">
   <label>Statement</label>
   <textarea id="statement" placeholder="e.g. Our state added 50,000 jobs last quarter."></textarea>
   <div class="row">
     <div><label>Speaker affiliation (optional)</label><input id="aff" placeholder="democrat / republican / none"></div>
     <div><label>Context (optional)</label><input id="ctx" placeholder="a press conference"></div>
   </div>
   <button id="go" onclick="verify()">Verify</button>
   <div class="ex">Try:
     <a onclick="ex('There are more American jobs in solar than in coal mining.','none','a campaign rally')">a real-ish claim</a> ·
     <a onclick="ex('The earth is flat and vaccines contain microchips.','none','a social media post')">an obvious falsehood</a>
   </div>
 </div>
 <div id="result" class="card hide"></div>
</div>
<script>
function ex(s,a,c){statement.value=s;aff.value=a;ctx.value=c}
async function verify(){
  const r=document.getElementById('result'), btn=document.getElementById('go');
  const s=document.getElementById('statement').value.trim();
  if(!s){return}
  btn.disabled=true; r.classList.remove('hide'); r.innerHTML='Verifying across the agents…';
  try{
    const resp=await fetch('api/verify',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({statement:s,speaker_affiliation:document.getElementById('aff').value,
        statement_context:document.getElementById('ctx').value})});
    const data=await resp.json();
    if(!resp.ok){throw new Error(data.error||('HTTP '+resp.status))}
    const v=data.results[0];
    const badge=v.prediction?'<span class="badge t">TRUE · truthful</span>':'<span class="badge f">FALSE · not truthful</span>';
    r.innerHTML=badge+
      '<div class="meta">Confidence '+(v.confidence*100).toFixed(0)+'% &nbsp;·&nbsp; '+
      'zero-shot agent: '+(v.agreement.zero_shot?'true':'false')+' &nbsp;·&nbsp; '+
      'fine-tuned agent: '+(v.agreement.fine_tuned?'true':'false')+'</div>'+
      '<div class="expl">'+v.explanation+'</div>';
  }catch(e){r.innerHTML='<span class="err">Error: '+e.message+'</span>'}
  finally{btn.disabled=false}
}
</script></body></html>"""


async def index(_request: Request):
    return HTMLResponse(PAGE)


async def api_verify(request: Request):
    body = await request.json()
    point = {
        "statement": body.get("statement", ""),
        "speaker_affiliation": body.get("speaker_affiliation") or None,
        "statement_context": body.get("statement_context") or None,
    }
    payload = {"points": [point]}
    headers = {"Content-Type": "application/json"}
    if API_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {API_AUTH_TOKEN}"
    async with httpx.AsyncClient(timeout=120) as hx:
        resp = await hx.post(f"{ORCHESTRATOR_URL}/verify", headers=headers, content=json.dumps(payload))
    return JSONResponse(resp.json(), status_code=resp.status_code)


async def health(_request: Request):
    return JSONResponse({"status": "ok", "role": "demo"})


def build_app() -> Starlette:
    return Starlette(routes=[
        Route("/", index, methods=["GET"]),
        Route("/api/verify", api_verify, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ])
