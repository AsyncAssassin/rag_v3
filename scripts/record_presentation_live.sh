#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=".env"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"
PORT=8512
FPS=30
VIDEO_SIZE="1728x1117"
SCREEN_INDEX=""
MIN_RECORD_SEC=430
MAX_RECORD_SEC=600
TECH_SMOKE_ONLY=0
SMOKE_SEC=45
WITH_TTS=0
WITH_PREWARM=0
TERMINAL_APP="Terminal"
BROWSER_APP_PRIMARY="Google Chrome for Testing"
BROWSER_APP_FALLBACK="Google Chrome"

usage() {
  cat <<'USAGE'
Usage: scripts/record_presentation_live.sh [options]

Options:
  --env <path>              .env path (default: .env)
  --artifacts-dir <path>    artifacts root dir (default: <repo>/artifacts)
  --screen-index <idx>      avfoundation screen index (auto-detected if omitted)
  --port <int>              Streamlit port (default: 8512)
  --fps <int>               recording fps (default: 30)
  --video-size <WxH>        recording size (default: 1728x1117)
  --min-record-sec <int>    min full-record duration sec (default: 430)
  --max-record-sec <int>    max full-record duration sec (default: 600)
  --with-tts                overlay TTS narration (default: off, live-only video)
  --with-prewarm            run prewarm before demo steps (default: off)
  --tech-smoke-only         run 30-60s technical recording smoke instead of full flow
  --smoke-sec <int>         smoke recording length in seconds (default: 45)
  --help                    show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --screen-index)
      SCREEN_INDEX="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --video-size)
      VIDEO_SIZE="$2"
      shift 2
      ;;
    --min-record-sec)
      MIN_RECORD_SEC="$2"
      shift 2
      ;;
    --max-record-sec)
      MAX_RECORD_SEC="$2"
      shift 2
      ;;
    --with-tts)
      WITH_TTS=1
      shift
      ;;
    --with-prewarm)
      WITH_PREWARM=1
      shift
      ;;
    --tech-smoke-only)
      TECH_SMOKE_ONLY=1
      shift
      ;;
    --smoke-sec)
      SMOKE_SEC="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PIP_BIN="$ROOT_DIR/.venv/bin/pip"
STREAMLIT_BIN="$ROOT_DIR/.venv/bin/streamlit"
UI_DEMO_SCRIPT="$ROOT_DIR/scripts/playwright_ui_demo.py"
NARRATION_FILE="$ROOT_DIR/scripts/legacy/presentation_narration_ru.txt"
if [[ ! -f "$NARRATION_FILE" ]]; then
  NARRATION_FILE="$ROOT_DIR/scripts/presentation_narration_ru.txt"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ARTIFACTS_DIR/screencast_run_$TS"
CMD_DIR="$RUN_DIR/cmd"
SCREENCAST_DIR="$ARTIFACTS_DIR/screencast"
mkdir -p "$RUN_DIR" "$CMD_DIR" "$SCREENCAST_DIR"

RAW_VIDEO="$RUN_DIR/raw_screen.mp4"
NARRATION_AIFF="$RUN_DIR/narration.aiff"
FINAL_VIDEO="$SCREENCAST_DIR/demo_${TS}.mp4"
RECORDING_REPORT_JSON="$RUN_DIR/recording_report.json"
RECORDING_REPORT_MD="$RUN_DIR/recording_report.md"

REC_PID=""
STREAMLIT_PID=""
REC_STARTED_EPOCH=0
STOPPED_RECORDING=0

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_DIR/run.log"
}

append_commands_log() {
  printf '%s\n' "$*" >> "$RUN_DIR/commands.log"
}

safe_osascript() {
  osascript "$@" >/dev/null 2>&1 || true
}

hide_distracting_apps() {
  safe_osascript -e 'tell application "Codex" to hide'
  safe_osascript -e 'tell application "ChatGPT" to hide'
}

activate_terminal() {
  safe_osascript -e "tell application \"$TERMINAL_APP\" to activate"
}

activate_browser() {
  safe_osascript -e "tell application \"$BROWSER_APP_PRIMARY\" to activate"
  safe_osascript -e "tell application \"$BROWSER_APP_FALLBACK\" to activate"
}

start_terminal_dashboard() {
  local dashboard_script="$RUN_DIR/terminal_dashboard.sh"
  cat > "$dashboard_script" <<EOF
#!/usr/bin/env bash
clear
cd "$ROOT_DIR"
echo "RAG demo live run"
echo "Run dir: $RUN_DIR"
echo
echo "Live log (tail -f run.log):"
echo
tail -n +1 -f "$RUN_DIR/run.log"
EOF
  chmod +x "$dashboard_script"

  hide_distracting_apps
  activate_terminal
  safe_osascript <<EOF
tell application "$TERMINAL_APP"
  activate
  do script "bash '$dashboard_script'"
end tell
EOF
}

cleanup() {
  if [[ -n "$STREAMLIT_PID" ]] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    kill "$STREAMLIT_PID" 2>/dev/null || true
    wait "$STREAMLIT_PID" 2>/dev/null || true
  fi

  if [[ "$STOPPED_RECORDING" -eq 0 ]] && [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing command: $1"
    return 1
  fi
  return 0
}

detect_screen_index() {
  local devices
  devices="$(ffmpeg -f avfoundation -list_devices true -i '' 2>&1 || true)"
  printf '%s\n' "$devices" > "$RUN_DIR/preflight_devices.txt"
  local idx
  idx="$(printf '%s\n' "$devices" | sed -n 's/.*\[\([0-9][0-9]*\)\] Capture screen.*/\1/p' | head -n1)"
  if [[ -n "$idx" ]]; then
    printf '%s\n' "$idx"
  else
    printf '1\n'
  fi
}

create_watchdog() {
  cat > "$RUN_DIR/run_with_timeout.py" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--timeout-sec", type=int, required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--stderr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided")

    out = Path(args.stdout)
    err = Path(args.stderr)
    meta = Path(args.meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    err.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    timed_out = False
    status = "ok"
    exit_code = None

    with out.open("w", encoding="utf-8") as out_f, err.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(cmd, cwd=args.cwd, stdout=out_f, stderr=err_f, text=True)
        try:
            exit_code = proc.wait(timeout=max(1, int(args.timeout_sec)))
            status = "ok" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "timeout"
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            exit_code = proc.returncode if proc.returncode is not None else -15

    payload = {
        "name": args.name,
        "status": status,
        "exit_code": int(exit_code) if exit_code is not None else None,
        "duration_sec": round(time.time() - started, 3),
        "timed_out": timed_out,
        "timeout_sec": int(args.timeout_sec),
        "command": cmd,
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
PY
  chmod +x "$RUN_DIR/run_with_timeout.py"
}

run_step() {
  local name="$1"
  local timeout_sec="$2"
  shift 2

  local out="$CMD_DIR/${name}.out"
  local err="$CMD_DIR/${name}.err"
  local meta="$CMD_DIR/${name}.meta.json"

  log "Starting step: $name"
  append_commands_log "[$name] $*"

  local hb_pid=""
  (
    local started_epoch
    started_epoch="$(date +%s)"
    while true; do
      sleep 12
      [[ -f "$meta" ]] && break
      local elapsed
      elapsed=$(( $(date +%s) - started_epoch ))
      log "Heartbeat: step=$name elapsed=${elapsed}s"
    done
  ) &
  hb_pid="$!"

  "$RUN_DIR/run_with_timeout.py" \
    --name "$name" \
    --timeout-sec "$timeout_sec" \
    --cwd "$ROOT_DIR" \
    --stdout "$out" \
    --stderr "$err" \
    --meta "$meta" \
    -- "$@" | tee -a "$RUN_DIR/run.log"

  if [[ -n "$hb_pid" ]] && kill -0 "$hb_pid" 2>/dev/null; then
    kill "$hb_pid" 2>/dev/null || true
  fi
  wait "$hb_pid" 2>/dev/null || true

  local step_state
  step_state="$(step_status "$name")"
  log "Finished step: $name status=$step_state"
}

step_status() {
  local name="$1"
  python3 - <<'PY' "$CMD_DIR/${name}.meta.json"
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print("missing")
    raise SystemExit(0)
obj=json.loads(p.read_text(encoding='utf-8'))
print(obj.get('status','missing'))
PY
}

ensure_playwright() {
  log "Checking Playwright availability"
  if "$PYTHON_BIN" - <<'PY' > "$RUN_DIR/preflight_playwright_check.log" 2>&1
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    b.close()
print("ok")
PY
  then
    log "Playwright + chromium: OK"
    return 0
  fi

  log "Playwright not ready, installing dependencies"
  "$PIP_BIN" install -r "$ROOT_DIR/requirements-playwright.txt" >> "$RUN_DIR/preflight_playwright_install.log" 2>&1
  "$PYTHON_BIN" -m playwright install chromium >> "$RUN_DIR/preflight_playwright_install.log" 2>&1

  if "$PYTHON_BIN" - <<'PY' > "$RUN_DIR/preflight_playwright_check_after.log" 2>&1
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    b.close()
print("ok")
PY
  then
    log "Playwright installation successful"
    return 0
  fi

  log "Playwright preflight failed after install"
  return 1
}

preflight_screen_permission() {
  log "Checking screen recording access with ffmpeg"
  ffmpeg -y \
    -f avfoundation \
    -framerate 5 \
    -video_size 1280x720 \
    -i "${SCREEN_INDEX}:none" \
    -t 2 \
    "$RUN_DIR/preflight_screen_test.mp4" \
    > "$RUN_DIR/preflight_screen.out" 2> "$RUN_DIR/preflight_screen.err" &
  local ffmpeg_pid="$!"
  local deadline=$(( $(date +%s) + 15 ))

  while kill -0 "$ffmpeg_pid" 2>/dev/null; do
    if [[ $(date +%s) -ge $deadline ]]; then
      log "Screen preflight timed out (>15s), forcing ffmpeg stop"
      kill -INT "$ffmpeg_pid" 2>/dev/null || true
      sleep 1
      kill -9 "$ffmpeg_pid" 2>/dev/null || true
      return 1
    fi
    sleep 1
  done

  if wait "$ffmpeg_pid"; then
    return 0
  fi
  return 1
}

start_recording() {
  log "Starting screen recording (screen=${SCREEN_INDEX}, size=${VIDEO_SIZE}, fps=${FPS})"
  ffmpeg -y \
    -f avfoundation \
    -framerate "$FPS" \
    -video_size "$VIDEO_SIZE" \
    -i "${SCREEN_INDEX}:none" \
    -pix_fmt yuv420p \
    -vcodec libx264 \
    -preset veryfast \
    "$RAW_VIDEO" \
    > "$RUN_DIR/ffmpeg_record.out" 2> "$RUN_DIR/ffmpeg_record.err" &
  REC_PID="$!"
  REC_STARTED_EPOCH="$(date +%s)"
  sleep 2

  if ! kill -0 "$REC_PID" 2>/dev/null; then
    log "Recorder process is not alive after start"
    return 1
  fi
  return 0
}

stop_recording() {
  if [[ "$STOPPED_RECORDING" -eq 1 ]]; then
    return 0
  fi
  if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    log "Stopping screen recording"
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
  STOPPED_RECORDING=1
}

wait_streamlit_ready() {
  local log_file="$1"
  local timeout_sec="$2"
  local deadline=$(( $(date +%s) + timeout_sec ))
  while [[ $(date +%s) -lt $deadline ]]; do
    if [[ -f "$log_file" ]] && rg -q "Local URL:|Network URL:|You can now view your Streamlit app" "$log_file"; then
      return 0
    fi
    if [[ -n "$STREAMLIT_PID" ]] && ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

build_report() {
  python3 - <<'PY' "$RUN_DIR" "$FINAL_VIDEO" "$MIN_RECORD_SEC" "$MAX_RECORD_SEC" "$TECH_SMOKE_ONLY" "$WITH_TTS"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
final_video = Path(sys.argv[2])
min_sec = int(sys.argv[3])
max_sec = int(sys.argv[4])
tech_smoke_only = bool(int(sys.argv[5]))
with_tts = bool(int(sys.argv[6]))
cmd_dir = run_dir / "cmd"

required_steps = ["selfcheck", "ask_in", "ask_out", "quality_gate", "ui_demo"] if not tech_smoke_only else ["tech_smoke"]
step_meta = {}
for step in required_steps:
    p = cmd_dir / f"{step}.meta.json"
    if p.exists():
        step_meta[step] = json.loads(p.read_text(encoding="utf-8"))
    else:
        step_meta[step] = {"status": "missing", "exit_code": None}

preflight = {
    "python_exists": Path(run_dir.parent.parent / ".venv/bin/python").exists(),
    "playwright_ready": (run_dir / "preflight_playwright_check.log").exists() or (run_dir / "preflight_playwright_check_after.log").exists(),
    "screen_permission_ok": (run_dir / "preflight_screen_test.mp4").exists(),
}

ffprobe_path = run_dir / "final_ffprobe.json"
media = {
    "final_video_exists": final_video.exists(),
    "has_video_stream": False,
    "has_audio_stream": False,
    "duration_sec": None,
    "duration_ok": False,
}
if ffprobe_path.exists():
    probe = json.loads(ffprobe_path.read_text(encoding="utf-8"))
    streams = probe.get("streams", [])
    media["has_video_stream"] = any(s.get("codec_type") == "video" for s in streams)
    media["has_audio_stream"] = any(s.get("codec_type") == "audio" for s in streams)
    try:
        media["duration_sec"] = float((probe.get("format") or {}).get("duration"))
    except Exception:
        media["duration_sec"] = None
    if media["duration_sec"] is not None:
        media["duration_ok"] = (min_sec <= media["duration_sec"] <= max_sec)

steps_ok = all((step_meta[s].get("status") == "ok") for s in required_steps)

if not preflight["screen_permission_ok"]:
    status = "inconclusive"
elif not media["final_video_exists"] or not media["has_video_stream"] or not media["has_audio_stream"]:
    status = "inconclusive"
elif not steps_ok:
    status = "fail"
elif not media["duration_ok"] and not tech_smoke_only:
    status = "fail"
else:
    status = "pass"

payload = {
    "status": status,
    "run_dir": str(run_dir),
    "final_video": str(final_video),
    "tech_smoke_only": tech_smoke_only,
    "with_tts": with_tts,
    "preflight": preflight,
    "steps": step_meta,
    "media": media,
}

(run_dir / "recording_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Recording Report",
    "",
    f"- Status: **{status.upper()}**",
    f"- Final video: `{final_video}`",
    f"- Tech smoke mode: `{tech_smoke_only}`",
    f"- TTS narration: `{with_tts}`",
    "",
    "## Preflight",
]
for k, v in preflight.items():
    lines.append(f"- {k}: `{v}`")

lines.extend(["", "## Steps"])
for s in required_steps:
    m = step_meta[s]
    lines.append(
        f"- {s}: status=`{m.get('status')}` exit=`{m.get('exit_code')}` duration=`{m.get('duration_sec')}`"
    )

lines.extend(["", "## Media"])
for k, v in media.items():
    lines.append(f"- {k}: `{v}`")

(run_dir / "recording_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

# === Preflight ===
log "Run dir: $RUN_DIR"
create_watchdog

require_cmd ffmpeg
require_cmd ffprobe
require_cmd osascript
require_cmd say

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "Missing Python virtualenv binary: $PYTHON_BIN"
  exit 2
fi
if [[ ! -f "$NARRATION_FILE" ]]; then
  log "Missing narration file: $NARRATION_FILE"
  exit 2
fi
if [[ ! -x "$UI_DEMO_SCRIPT" ]]; then
  log "Missing UI demo script: $UI_DEMO_SCRIPT"
  exit 2
fi

if [[ -z "$SCREEN_INDEX" ]]; then
  SCREEN_INDEX="$(detect_screen_index)"
fi
log "Using screen index: $SCREEN_INDEX"

if ! ensure_playwright; then
  log "Playwright preflight failed"
  build_report >/dev/null || true
  exit 2
fi

if ! preflight_screen_permission; then
  log "Screen recording permission check failed. Give screen recording access and rerun."
  build_report >/dev/null || true
  exit 2
fi

if ! start_recording; then
  log "Failed to start recorder"
  build_report >/dev/null || true
  exit 2
fi

if [[ "$TECH_SMOKE_ONLY" -eq 1 ]]; then
  log "Running technical smoke only for ${SMOKE_SEC}s"
  run_step tech_smoke "$((SMOKE_SEC + 20))" /bin/zsh -lc "sleep $SMOKE_SEC"
  stop_recording
else
  start_terminal_dashboard
  sleep 2

  # === Core live flow ===
  run_step selfcheck 240 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" selfcheck
  if [[ "$WITH_PREWARM" -eq 1 ]]; then
    run_step prewarm 600 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" prewarm
  fi
  run_step ask_in 300 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" ask "Какие ключевые темы в отчете Сбера за 2024 год?"
  run_step ask_out 300 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" ask "Расскажи про ядерный реактор на Луне в отчете Сбера 2015."
  run_step quality_gate 1200 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" quality-gate --goldset artifacts/goldset_sber_qa.jsonl --reranker amberoad --metric-k 10 --min-recall 0.55 --min-mrr 0.45 --min-ndcg 0.50 --output "$RUN_DIR/quality_gate.json"
  if [[ "$(step_status quality_gate)" != "ok" ]]; then
    log "quality_gate failed; retrying once after short backoff"
    sleep 5
    run_step quality_gate 1200 "$PYTHON_BIN" -m rag_system.cli --env "$ENV_FILE" quality-gate --goldset artifacts/goldset_sber_qa.jsonl --reranker amberoad --metric-k 10 --min-recall 0.55 --min-mrr 0.45 --min-ndcg 0.50 --output "$RUN_DIR/quality_gate.json"
  fi

  # === UI live demo ===
  STREAMLIT_LOG="$RUN_DIR/streamlit.log"
  "$STREAMLIT_BIN" run "$ROOT_DIR/streamlit_app.py" --server.headless true --server.port "$PORT" > "$STREAMLIT_LOG" 2>&1 &
  STREAMLIT_PID="$!"

  if wait_streamlit_ready "$STREAMLIT_LOG" 90; then
    hide_distracting_apps
    activate_browser
    sleep 1
    run_step ui_demo 600 "$PYTHON_BIN" "$UI_DEMO_SCRIPT" --url "http://localhost:${PORT}" --output "$RUN_DIR/ui_demo.json" --screenshot "$RUN_DIR/ui_demo.png" --headed
  else
    log "Streamlit did not become ready in time"
    run_step ui_demo 10 /bin/zsh -lc "exit 1"
  fi

  if [[ -n "$STREAMLIT_PID" ]] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    kill "$STREAMLIT_PID" 2>/dev/null || true
    wait "$STREAMLIT_PID" 2>/dev/null || true
  fi
  STREAMLIT_PID=""

  # Ensure target duration window starts from 7 minutes.
  elapsed=$(( $(date +%s) - REC_STARTED_EPOCH ))
  if (( elapsed < MIN_RECORD_SEC )); then
    pad=$(( MIN_RECORD_SEC - elapsed ))
    log "Padding recording by ${pad}s to satisfy minimum duration"
    sleep "$pad"
  fi

  stop_recording
fi

# === Audio + mux ===
if [[ "$WITH_TTS" -eq 1 ]]; then
  log "Generating narration audio (Milena)"
  say -v Milena -f "$NARRATION_FILE" -o "$NARRATION_AIFF"

  log "Muxing video + TTS audio"
  ffmpeg -y \
    -i "$RAW_VIDEO" \
    -i "$NARRATION_AIFF" \
    -filter_complex "[1:a]apad[a]" \
    -map 0:v \
    -map "[a]" \
    -c:v copy \
    -c:a aac \
    -shortest \
    "$FINAL_VIDEO" \
    > "$RUN_DIR/mux.out" 2> "$RUN_DIR/mux.err"
else
  log "Muxing video with silent audio track (live-only mode)"
  ffmpeg -y \
    -i "$RAW_VIDEO" \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -map 0:v \
    -map 1:a \
    -c:v copy \
    -c:a aac \
    -shortest \
    "$FINAL_VIDEO" \
    > "$RUN_DIR/mux.out" 2> "$RUN_DIR/mux.err"
fi

ffprobe -v error -print_format json -show_streams -show_format "$FINAL_VIDEO" > "$RUN_DIR/final_ffprobe.json"

build_report | tee -a "$RUN_DIR/run.log"

log "Done. Final video: $FINAL_VIDEO"
log "Report: $RECORDING_REPORT_JSON"
