"""결과 웹 페이지 (표준 라이브러리 http.server만 사용). scripts/serve_results.py가 실행한다.

  /                         보기 페이지 (주행 선택, 결과 지도, 구역 표, 요약, 평가, 처리 시간)
  /api/runs                 results/ 아래 정찰 결과 목록
  /api/run/<run>            한 주행의 요약 (dirt_regions.json + evaluation.json + timing 중앙값)
  /files/<run>/<경로>       결과 파일 (dirt_map.png, regions/*.jpg …)

읽기 전용이며 results/ 폴더 밖의 파일은 내주지 않는다.
정찰이 진행 중이면 run_patrol.py가 몇 초마다 지도·json을 바꿔 쓰므로, 페이지가 2초마다 다시 불러온다.
"""

import csv
import json
import mimetypes
import os
import statistics
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


def list_runs(root):
    """정찰 결과(dirt_regions.json이 있는 폴더) 목록, 최근 것부터."""
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in os.listdir(root):
        d = os.path.join(root, name)
        j = os.path.join(d, "dirt_regions.json")
        if os.path.isfile(j):
            try:
                with open(j, "r", encoding="utf-8") as f:
                    n = json.load(f).get("n_regions")
            except (OSError, ValueError):
                n = None
            runs.append({"name": name, "mtime": os.path.getmtime(j), "n_regions": n,
                         "evaluated": os.path.isfile(os.path.join(d, "evaluation.json"))})
    runs.sort(key=lambda r: -r["mtime"])
    return runs


def timing_medians(path):
    """timing.csv → 단계별 중앙값 [ms]."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        if k == "frame_id":
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                pass
        if vals:
            out[k[:-3] if k.endswith("_ms") else k] = round(statistics.median(vals), 2)
    out["frames"] = len(rows)
    return out


def run_summary(root, run):
    d = os.path.join(root, run)
    with open(os.path.join(d, "dirt_regions.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    ev = os.path.join(d, "evaluation.json")
    if os.path.isfile(ev):
        with open(ev, "r", encoding="utf-8") as f:
            data["evaluation"] = json.load(f)
    data["timing"] = timing_medians(os.path.join(d, "timing.csv"))
    rv = os.path.join(d, "revisit.json")
    if os.path.isfile(rv):
        with open(rv, "r", encoding="utf-8") as f:
            data["revisit"] = json.load(f)
    from floor_dirt.tegralog import summarize_file
    data["resources"] = summarize_file(os.path.join(d, "tegrastats.log")) if os.path.isfile(
        os.path.join(d, "tegrastats.log")) else None
    data["map_mtime"] = os.path.getmtime(os.path.join(d, "dirt_map.png")) if os.path.isfile(
        os.path.join(d, "dirt_map.png")) else None
    return data


def safe_path(root, rel):
    """root 안의 실제 경로. root 밖(../ 등)이면 None."""
    root_real = os.path.realpath(root)
    p = os.path.realpath(os.path.join(root_real, rel))
    if p != root_real and not p.startswith(root_real + os.sep):
        return None
    return p


class ResultsServer:
    def __init__(self, results_dir, port=8000, host="0.0.0.0"):
        self.root = os.path.abspath(results_dir)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code=200):
                self._send(code, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

            def do_GET(self):
                path = unquote(self.path.split("?", 1)[0])
                if path in ("/", "/index.html"):
                    return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
                if path == "/api/runs":
                    return self._json(list_runs(server.root))
                if path.startswith("/api/run/"):
                    run = path[len("/api/run/"):].strip("/")
                    d = safe_path(server.root, run)
                    if d is None or "/" in run or not os.path.isfile(os.path.join(d, "dirt_regions.json")):
                        return self._json({"error": "없는 주행: {}".format(run)}, 404)
                    try:
                        return self._json(run_summary(server.root, run))
                    except (OSError, ValueError) as e:   # 쓰는 도중에 읽은 경우 등
                        return self._json({"error": str(e)}, 503)
                if path.startswith("/files/"):
                    p = safe_path(server.root, path[len("/files/"):])
                    if p is None or not os.path.isfile(p):
                        return self._send(404, "text/plain; charset=utf-8", b"not found")
                    with open(p, "rb") as f:
                        body = f.read()
                    ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
                    return self._send(200, ctype, body)
                self._send(404, "text/plain; charset=utf-8", b"not found")

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self):
        self.httpd.serve_forever()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>청소 구역 결과</title>
<style>
  :root { --bg:#f6f6f4; --card:#fff; --ink:#222; --dim:#777; --line:#e2e2de; --accent:#b0209a; --bad:#d23c3c; --ok:#2e8b57; }
  @media (prefers-color-scheme: dark) { :root { --bg:#181818; --card:#232323; --ink:#e6e6e6; --dim:#9a9a9a; --line:#333; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family: system-ui, "Malgun Gothic", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0 12px 0 0; }
  select, button { font-size:14px; padding:4px 8px; }
  #status { color:var(--dim); font-size:13px; }
  main { display:grid; grid-template-columns: minmax(0, 3fr) minmax(280px, 2fr); gap:16px; padding:16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:16px; }
  .card h2 { font-size:15px; margin:0 0 8px; }
  #map { width:100%; height:auto; image-rendering: pixelated; border-radius:4px; background:#fff; }
  .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:8px; }
  .stat { border:1px solid var(--line); border-radius:6px; padding:8px; }
  .stat b { display:block; font-size:20px; }
  .stat span { color:var(--dim); font-size:12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { border-bottom:1px solid var(--line); padding:6px 4px; text-align:left; vertical-align:middle; }
  th { color:var(--dim); font-weight:normal; }
  td img { height:56px; border-radius:4px; }
  .num { font-variant-numeric: tabular-nums; }
  .empty { color:var(--dim); }
  .bad { color:var(--bad); } .ok { color:var(--ok); }
</style>
</head>
<body>
<header>
  <h1>청소 구역 탐지 결과</h1>
  <label>주행 <select id="run"></select></label>
  <label><input type="checkbox" id="live" checked> 자동 새로고침 (2초)</label>
  <span id="status">불러오는 중…</span>
</header>
<main>
  <section>
    <div class="card"><h2>오염 지도</h2><img id="map" alt="오염 지도">
      <div style="font-size:12px;color:var(--dim);margin-top:6px">
        연두 = 깨끗함, 노랑→빨강 = 오염 의심(확률), 회색 빗금 = 봤지만 판정 못 함, 자홍 사각형 = 오염 구역(번호), 파랑 = 로봇·경로
      </div></div>
  </section>
  <section>
    <div class="card"><h2>요약</h2><div class="stats" id="stats"></div></div>
    <div class="card"><h2>오염 구역</h2><div id="regions"></div></div>
    <div class="card" id="rvcard" hidden><h2>재방문 검증</h2><div id="rv"></div></div>
    <div class="card" id="evalcard" hidden><h2>평가 (정답과 비교)</h2><div id="eval"></div></div>
    <div class="card"><h2>프레임당 처리 시간 (중앙값)</h2><div id="timing"></div></div>
    <div class="card" id="rescard" hidden><h2>젯슨 자원 사용 (tegrastats)</h2><div id="res"></div></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let current = null, lastMap = null;
const fmt = (v, d=2) => (v === null || v === undefined) ? "-" : Number(v).toFixed(d);
const cm = (m) => (m === null || m === undefined) ? "-" : (m * 100).toFixed(1);

async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  const sel = $("run"), keep = sel.value;
  sel.innerHTML = runs.map(r => `<option value="${r.name}">${r.name} (구역 ${r.n_regions ?? "?"})</option>`).join("");
  if (!runs.length) { $("status").textContent = "results/ 에 정찰 결과가 없습니다"; return; }
  sel.value = runs.some(r => r.name === keep) ? keep : runs[0].name;
  if (sel.value !== current) { current = sel.value; lastMap = null; refresh(); }
}

function stat(label, value) { return `<div class="stat"><b class="num">${value}</b><span>${label}</span></div>`; }

async function refresh() {
  if (!current) return;
  let d;
  try { d = await (await fetch(`/api/run/${encodeURIComponent(current)}`)).json(); }
  catch (e) { $("status").textContent = "연결 끊김"; return; }
  if (d.error) { $("status").textContent = d.error; return; }
  if (d.map_mtime !== lastMap) { $("map").src = `/files/${encodeURIComponent(current)}/dirt_map.png?t=${d.map_mtime}`; lastMap = d.map_mtime; }
  const s = d.summary || {};
  $("stats").innerHTML = stat("오염 구역", d.n_regions) + stat("의심 넓이 m²", fmt(s.suspect_m2, 3)) +
    stat("깨끗함 m²", fmt(s.clean_m2)) + stat("확인 불가 m²", fmt(s.unknown_m2)) +
    stat("처리 프레임", d.frames ?? "-") + stat("조명 변화 (밝기)", fmt(d.light_offset_median, 1));
  const photos = d.photos || {};
  $("regions").innerHTML = d.regions.length ? `<table><tr><th>#</th><th>중심 x, y [m]</th><th>크기 [cm]</th><th>넓이 [cm²]</th><th>확률</th><th>사진</th></tr>` +
    d.regions.map(r => `<tr><td>${r.id}</td><td class="num">${fmt(r.center[0])}, ${fmt(r.center[1])}</td>
      <td class="num">${cm(r.size[0])} × ${cm(r.size[1])}</td><td class="num">${(r.area_m2 * 1e4).toFixed(0)}</td>
      <td class="num">${fmt(r.max_p)}</td>
      <td>${photos[r.id] ? `<a href="/files/${encodeURIComponent(current)}/${photos[r.id]}" target="_blank"><img src="/files/${encodeURIComponent(current)}/${photos[r.id]}"></a>` : "-"}</td></tr>`).join("") + `</table>`
    : `<p class="empty">아직 오염 구역이 없습니다</p>`;
  const rv = d.revisit;
  $("rvcard").hidden = !rv;
  if (rv) {
    const cls = { cleaned: "ok", needs_cleaning: "bad", unverified: "" };
    $("rv").innerHTML = `<div class="stats">` + stat("처리 완료", rv.summary.cleaned) + stat("재청소 필요", rv.summary.needs_cleaning) +
      stat("확인 불가", rv.summary.unverified) + `</div><table style="margin-top:8px"><tr><th>구역</th><th>판정</th><th>복원율</th><th>본 비율</th><th>사진</th></tr>` +
      rv.regions.map(r => `<tr><td>${r.id}</td><td class="${cls[r.status]}">${r.status_ko}</td>
        <td class="num">${r.restoration_ratio === null ? "-" : Math.round(r.restoration_ratio * 100) + "%"}</td>
        <td class="num">${Math.round(r.coverage * 100)}%</td><td class="num">${r.frames}</td></tr>`).join("") + `</table>`;
  }
  const ev = d.evaluation;
  $("evalcard").hidden = !ev;
  if (ev) {
    $("eval").innerHTML = `<div class="stats">` + stat("재현율", ev.recall === null ? "-" : Math.round(ev.recall * 100) + "%") +
      stat("오탐 구역", ev.false_positives) + stat("위치 오차 평균 cm", cm(ev.error_mean_m)) + stat("최대 cm", cm(ev.error_max_m)) + `</div>` +
      `<table style="margin-top:8px"><tr><th>정답</th><th>위치 [m]</th><th>검출</th><th>구역</th><th>오차 [cm]</th></tr>` +
      ev.items.map(it => `<tr><td>${it.name}</td><td class="num">${fmt(it.x)}, ${fmt(it.y)}</td>
        <td class="${it.found ? "ok" : "bad"}">${it.found ? "O" : "X"}</td><td>${it.region ?? "-"}</td><td class="num">${cm(it.error_m)}</td></tr>`).join("") + `</table>`;
  }
  const t = d.timing;
  $("timing").innerHTML = t ? `<table>` + Object.entries(t).filter(([k]) => k !== "frames")
      .map(([k, v]) => `<tr><td>${k === "total" ? "<b>합계</b>" : k}</td><td class="num">${v} ms</td></tr>`).join("") + `</table>`
    : `<p class="empty">timing.csv 없음</p>`;
  const rs = d.resources;
  $("rescard").hidden = !rs;
  if (rs) {
    const temps = Object.entries(rs.temp_max_c || {}).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k, v]) => `${k} ${v.toFixed(1)}°C`).join(", ");
    $("res").innerHTML = `<div class="stats">` + stat("메모리 최대 MB", `${rs.ram_max_mb} / ${rs.ram_total_mb}`) +
      stat("CPU 평균 %", fmt(rs.cpu_mean_pct, 0)) + stat("GPU 평균 %", fmt(rs.gpu_mean_pct, 0)) +
      stat("GPU 최대 %", rs.gpu_max_pct ?? "-") + `</div><p style="font-size:12px;color:var(--dim)">최고 온도: ${temps || "-"} (${rs.samples}초 기록)</p>`;
  }
  $("status").textContent = `갱신 ${new Date().toLocaleTimeString()}`;
}

$("run").onchange = () => { current = $("run").value; lastMap = null; refresh(); };
loadRuns();
setInterval(() => { if ($("live").checked) refresh(); }, 2000);
setInterval(() => { if ($("live").checked) loadRuns(); }, 10000);
</script>
</body>
</html>
"""
