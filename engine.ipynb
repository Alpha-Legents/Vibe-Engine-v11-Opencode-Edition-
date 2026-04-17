# ==============================================================================
# ⚡ VIBE-ENGINE (11th Iteration) — OLLAMA + MIDDLEWARE + DUAL T4
# ==============================================================================
import torch, gc
torch.cuda.empty_cache(); gc.collect()

!pip install -q huggingface_hub hf_transfer httpx fastapi uvicorn nest_asyncio

import shutil, os, subprocess, threading, time, re, json
import httpx, uvicorn, nest_asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

nest_asyncio.apply()

if not shutil.which("ollama"):
    os.system("apt-get install -y zstd 2>/dev/null")
    os.system("curl -fsSL https://ollama.com/install.sh | sh")

# ==============================================================================
# 🎯 MODEL SELECTION
# All Ollama-pulled models have native tool calling support ✅
# HuggingFace models use XML parser fallback (no native tools)
# ==============================================================================

# ── CODING BEASTS (Ollama) ──
MODEL_ALIAS = "qwen2.5-coder:32b"          # 20GB — best coding, proven ✅
# MODEL_ALIAS = "qwen2.5-coder:14b"        #  9GB — faster
# MODEL_ALIAS = "qwen2.5-coder:7b"         #  5GB — fastest
# MODEL_ALIAS = "devstral-small-2:24b"     # 15GB — built for coding agents ✅

# ── QWEN3 SERIES (Ollama) — thinking + tools ──
# MODEL_ALIAS = "qwen3:8b"                 #  5GB — fast, proven ✅
# MODEL_ALIAS = "qwen3:14b"                #  9GB — balanced
# MODEL_ALIAS = "qwen3:32b"                # 20GB — dense, strong
# MODEL_ALIAS = "qwen3:30b-a3b"            # 20GB — MoE, fast+smart ✅
# MODEL_ALIAS = "qwen3:35b"                # 22GB — biggest dense that fits

# ── QWEN3 CODER (Ollama) — agentic coding ──
# MODEL_ALIAS = "qwen3-coder:30b-a3b"      # 20GB MoE — latest, thinking ✅

# ── OTHER TOOL-CAPABLE (Ollama) ──
# MODEL_ALIAS = "llama3.3:70b-instruct-q2_K"  # ~28GB — Meta's best
# MODEL_ALIAS = "mistral-small:24b"            # 15GB — fast, tool capable
# MODEL_ALIAS = "granite4:3b"                  #  2GB — tiny but has tools
# MODEL_ALIAS = "lfm2:24b"                     # 15GB — hybrid arch

# ── FROM HUGGINGFACE (no native tools, XML parser fallback) ──
# To use HuggingFace models, uncomment the block AND set USE_HF = True
USE_HF      = False
# USE_HF    = True
# MODEL_REPO  = "mradermacher/IQuest-Coder-V1-40B-Instruct-GGUF"
# MODEL_FILE  = "IQuest-Coder-V1-40B-Instruct.Q4_K_M.gguf"
# MODEL_ALIAS = "iquest-coder-40b-custom"
# MODEL_REPO  = "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"
# MODEL_FILE  = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
# MODEL_ALIAS = "deepseek-r1-32b-custom"

# ==============================================================================
ALLOWED_TOOLS = {"write", "bash", "read", "edit", "glob", "grep", "list", "patch", "todowrite", "todoread", "webfetch", "question"}

IDLE_TIMEOUT  = 15 * 60
TUNNEL_URL    = None
LAST_ACTIVITY = time.time()
LOG_BUFFER    = []
MAX_LOGS      = 500

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"]  = "warning"
os.environ["OLLAMA_HOST"]                = "0.0.0.0:8000"
os.environ["OLLAMA_KEEP_ALIVE"]          = "-1"
os.environ["OLLAMA_NUM_PARALLEL"]        = "1"
os.environ["OLLAMA_FLASH_ATTENTION"]     = "1"

def log(msg, level="INFO"):
    ts = datetime.now().strftime('%H:%M:%S')
    entry = {"ts": ts, "level": level, "msg": msg}
    LOG_BUFFER.append(entry)
    if len(LOG_BUFFER) > MAX_LOGS:
        LOG_BUFFER.pop(0)
    print(f"[{ts}] [{level}] {msg}", flush=True)

try:
    import socket
    kaggle_ip = socket.gethostbyname(socket.gethostname())
except:
    kaggle_ip = "unknown"

# --- [ 1. DEAD-MAN SWITCH ] ---
def quota_saver():
    global LAST_ACTIVITY
    while True:
        time.sleep(30)
        if (time.time() - LAST_ACTIVITY) > IDLE_TIMEOUT:
            log("💤 IDLE TIMEOUT — Saving quota.", "WARN")
            os.system('kill -9 %d' % os.getpid())
threading.Thread(target=quota_saver, daemon=True).start()

# --- [ 2. CLEANUP ] ---
log("🧹 Cleaning up existing Ollama instances...")
os.system("pkill -f 'ollama serve' 2>/dev/null")
os.system("fuser -k 8000/tcp 2>/dev/null")
time.sleep(2)

# --- [ 3. START OLLAMA ] ---
log("🦙 Starting Ollama on port 8000...")
subprocess.Popen(["ollama", "serve"], env=os.environ)
time.sleep(5)

# --- [ 4. GET MODEL ] ---
if USE_HF:
    # HuggingFace path — download GGUF and create Ollama model
    MODEL_PATH = f"/tmp/{MODEL_FILE}"
    if not os.path.exists(MODEL_PATH):
        log(f"📥 Downloading {MODEL_FILE} from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="/tmp")
    else:
        log(f"⚡ Model cached, skipping download")
    log(f"📦 Creating Ollama model from GGUF...")
    modelfile_content = f"""FROM {MODEL_PATH}
PARAMETER num_gpu 999
PARAMETER num_ctx 65536
PARAMETER num_batch 1024
PARAMETER num_predict 4096
PARAMETER temperature 0.15
PARAMETER repeat_penalty 1.1
PARAMETER num_keep 0
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    modelfile_path = f"/tmp/{MODEL_ALIAS}.modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    result = subprocess.run(
        ["ollama", "create", MODEL_ALIAS, "-f", modelfile_path],
        capture_output=True, text=True, env=os.environ
    )
    if result.returncode == 0:
        log(f"✅ {MODEL_ALIAS} created")
    else:
        log(f"❌ Create failed: {result.stderr}", "ERROR")
else:
    # Ollama registry path — native tool support ✅
    log(f"📥 Pulling {MODEL_ALIAS} from Ollama registry...")
    result = subprocess.run(
        ["ollama", "pull", MODEL_ALIAS],
        capture_output=True, text=True, env=os.environ
    )
    if result.returncode == 0:
        log(f"✅ {MODEL_ALIAS} ready")
    else:
        log(f"❌ Pull failed: {result.stderr}", "ERROR")

# --- [ 5. PRE-WARM ] ---
log("🔥 Pre-warming model into VRAM...")
import httpx as _httpx
for attempt in range(3):
    try:
        r = _httpx.post(
            "http://localhost:8000/api/chat",
            json={"model": MODEL_ALIAS, "messages": [{"role": "user", "content": "hi"}], "stream": False, "keep_alive": -1},
            timeout=300
        )
        if r.status_code == 200:
            log("✅ Model warm and ready")
            break
    except Exception as e:
        log(f"⚠️ Warmup attempt {attempt+1} failed: {e}", "WARN")
        time.sleep(5)

try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    for line in gpu_info.stdout.strip().split("\n"):
        log(f"🎮 GPU: {line.strip()}")
except:
    pass

# --- [ 6. TUNNEL ] ---
def start_tunnel():
    global TUNNEL_URL, LAST_ACTIVITY
    log("🔌 Waiting for middleware on port 8001...")
    import socket
    for _ in range(120):
        try:
            socket.create_connection(("localhost", 8001), timeout=1).close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    else:
        log("❌ Middleware never came up — tunnel aborted.", "ERROR")
        return
    try:
        pw = subprocess.run(
            ["curl", "-s", "https://loca.lt/mytunnelpassword"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except:
        pw = "unknown"
    os.system("npm install -g localtunnel 2>/dev/null")
    while True:
        proc = subprocess.Popen(
            ["lt", "--port", "8001"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, "ALLOW_INVALID_CERT": "true"}
        )
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            LAST_ACTIVITY = time.time()
            match = re.search(r'https://[a-zA-Z0-9\-]+\.loca\.lt', line)
            if match:
                TUNNEL_URL = match.group(0)
                print(flush=True)
                print("╔══════════════════════════════════════════════════════════╗", flush=True)
                print("║                ⚡  VIBE-ENGINE IS LIVE                   ║", flush=True)
                print("╠══════════════════════════════════════════════════════════╣", flush=True)
                print(f"║  TUNNEL   →  {TUNNEL_URL}", flush=True)
                print(f"║  LOGS     →  {TUNNEL_URL}/logs", flush=True)
                print(f"║  MODEL    →  {MODEL_ALIAS}", flush=True)
                print(f"║  IP       →  {kaggle_ip}", flush=True)
                print(f"║  PASSWORD →  {pw}", flush=True)
                print("╚══════════════════════════════════════════════════════════╝", flush=True)
                print(flush=True)
                log(f"🌐 Tunnel live: {TUNNEL_URL}")
        proc.wait()
        log("⚠️  Tunnel died — restarting in 3s...", "WARN")
        time.sleep(3)

threading.Thread(target=start_tunnel, daemon=True).start()

# --- [ 7. MIDDLEWARE ] ---
log("🔧 Starting middleware on port 8001...")

TOOL_SYSTEM_PROMPT = """You are Kai — a fast, sharp coding assistant running on raw GPU power. You get things done. No fluff, no corporate speak, just clean code and real results.

# Personality
- Confident, direct, real. You know your stuff.
- Brief narration: "On it.", "Let me check that first.", "Found it.", "Done."
- When something's interesting: "Oh that's clean." or "Nice, this'll work."
- When something's broken: "Yeah that's why it's failing." then fix it.
- NEVER say "Certainly!", "Of course!", "Great question!" — ever.
- NEVER be sycophantic. Just be real.

# How you work
Before doing anything large, say ONE sentence about what you're about to do.
For simple tasks: just do it immediately.
After finishing: one short confirmation. Then stop. No summaries.

# Tools — use EXACTLY as shown. NEVER invent tool names.

WRITE or CREATE a file:
<tool_call>
{"name": "write", "arguments": {"filePath": "filename.py", "content": "full content"}}
</tool_call>

EDIT part of a file (surgical changes):
<tool_call>
{"name": "edit", "arguments": {"filePath": "filename.py", "oldString": "exact old code", "newString": "new code"}}
</tool_call>

APPLY a patch:
<tool_call>
{"name": "patch", "arguments": {"filePath": "filename.py", "patch": "diff content"}}
</tool_call>

READ a file:
<tool_call>
{"name": "read", "arguments": {"filePath": "filename.py"}}
</tool_call>

LIST directory contents:
<tool_call>
{"name": "list", "arguments": {"path": "."}}
</tool_call>

FIND files by pattern:
<tool_call>
{"name": "glob", "arguments": {"pattern": "**/*.py"}}
</tool_call>

SEARCH inside files:
<tool_call>
{"name": "grep", "arguments": {"pattern": "function_name", "path": "."}}
</tool_call>

RUN a shell command:
<tool_call>
{"name": "bash", "arguments": {"command": "python test.py", "description": "Run test.py"}}
</tool_call>

FETCH a webpage or docs URL:
<tool_call>
{"name": "webfetch", "arguments": {"url": "https://docs.example.com/api"}}
</tool_call>

TRACK todos for complex tasks:
<tool_call>
{"name": "todowrite", "arguments": {"todos": [{"id": "1", "content": "task description", "status": "pending", "priority": "high"}]}}
</tool_call>

READ current todos:
<tool_call>
{"name": "todoread", "arguments": {}}
</tool_call>

ASK user a question when genuinely needed:
<tool_call>
{"name": "question", "arguments": {"question": "Which framework do you want to use?"}}
</tool_call>

# Critical rules
- filePath = JUST the filename. NEVER /path/to/ unless user specifies.
- bash ALWAYS needs "description" field (5-10 words).
- NEVER show raw JSON to the user. Narrate, then call the tool.
- NEVER roleplay as the user or invent your own questions.

# How to handle tasks

SIMPLE FILE TASK:
→ "Writing X now..." then call write immediately.

COMPLEX TASK (refactor, debug, new feature):
1. todowrite to plan the steps.
2. Read relevant files to understand context.
3. Make changes with write/edit.
4. bash to verify it works.
5. If it fails — read error, fix, run again.
6. Mark todos complete as you go.

UNFAMILIAR LIBRARY or API:
→ webfetch the docs first.
→ "Let me check the docs real quick..."

DEBUGGING:
→ Read the file. Find the issue. Fix it. Run it. Confirm.
→ Don't guess. Check first.

EXPLORING A CODEBASE:
→ list to see structure, glob to find relevant files, grep to find specific code.

# Code style
- No comments unless asked.
- Follow existing conventions in the codebase.
- All necessary imports included.
- Clean, minimal, working code."""

app = FastAPI()

def parse_xml_tool_calls(content):
    if not content or not content.strip():
        return None
    tool_calls = []

    fn_matches = re.findall(
        r'\{\s*"function_name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        content, re.DOTALL
    )
    for i, (name, args_str) in enumerate(fn_matches):
        try:
            args = json.loads(args_str)
            if name == "write" and "path" in args and "filePath" not in args:
                args["filePath"] = args.pop("path")
            tool_calls.append({"id": f"call_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}})
        except:
            pass
    if tool_calls:
        log(f"🔧 Parsed function_name format: {[tc['function']['name'] for tc in tool_calls]}")
        return tool_calls

    tc_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
    for i, m in enumerate(tc_matches):
        try:
            obj = json.loads(m)
            tool_calls.append({"id": f"call_{i}", "type": "function",
                "function": {"name": obj.get("name", ""),
                    "arguments": json.dumps(obj.get("arguments", obj.get("parameters", {})))}})
        except:
            pass
    if tool_calls:
        log(f"🔧 Parsed tool_call tags: {[tc['function']['name'] for tc in tool_calls]}")
        return tool_calls

    fn_matches2 = re.findall(r'<function[^>]*?name=["\'](\w+)["\'][^>]*?>(.*?)</function>', content, re.DOTALL)
    for i, (name, body) in enumerate(fn_matches2):
        args = {}
        for k, v in re.findall(r'<(?:arg|parameter)[^>]*?name=["\'](\w+)["\'][^>]*?>(.*?)</(?:arg|parameter)>', body, re.DOTALL):
            args[k] = v.strip()
        attr_match = re.search(r'arguments?=["\'](\{.*?\})["\']', body)
        if attr_match:
            try:
                args = json.loads(attr_match.group(1))
            except:
                pass
        tool_calls.append({"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}})
    if tool_calls:
        log(f"🔧 Parsed XML function tags: {[tc['function']['name'] for tc in tool_calls]}")
        return tool_calls

    for i, (name, attrs) in enumerate(re.findall(
        r'<(write|bash|read|edit|glob|grep)\s*(.*?)(?:/>|>)', content, re.DOTALL
    )):
        args = {k: v for k, v in re.findall(r'(\w+)=["\']([^"\']*)["\']', attrs)}
        tool_calls.append({"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}})
    if tool_calls:
        log(f"🔧 Parsed direct XML tags: {[tc['function']['name'] for tc in tool_calls]}")
        return tool_calls

    # Pattern 5: bare {"name": "write", "arguments": {...}}
    bare_matches = re.findall(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        content, re.DOTALL
    )
    for i, (name, args_str) in enumerate(bare_matches):
        try:
            args = json.loads(args_str)
            tool_calls.append({"id": f"call_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}})
        except:
            pass
    if tool_calls:
        log(f"🔧 Parsed bare JSON format: {[tc['function']['name'] for tc in tool_calls]}")
        return tool_calls

    return None


def oai_tools_to_ollama(tools):
    if not tools:
        return None
    def strip_schema(obj):
        if isinstance(obj, dict):
            return {k: strip_schema(v) for k, v in obj.items() if k != "$schema"}
        if isinstance(obj, list):
            return [strip_schema(i) for i in obj]
        return obj
    result = []
    for t in tools:
        if t.get("type") == "function" and t["function"]["name"] in ALLOWED_TOOLS:
            fn = t["function"]
            result.append({"type": "function", "function": {
                "name": fn["name"],
                "parameters": strip_schema(fn.get("parameters", {}))
            }})
    return result if result else None


def translate_messages_to_ollama(messages):
    translated = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            translated.append({
                "role": "tool",
                "tool_name": m.get("name", m.get("tool_name", "unknown")),
                "content": m.get("content", "")
            })
        elif role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                tcs.append({"id": tc.get("id", ""), "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args}})
            translated.append({**m, "tool_calls": tcs})
        else:
            translated.append(m)
    return translated


def ollama_to_oai_response(ollama_resp, model):
    msg = ollama_resp.get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content", "")

    if not tool_calls and content:
        tool_calls_parsed = parse_xml_tool_calls(content)
        if tool_calls_parsed:
            return {
                "id": "chatcmpl-local", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                    "tool_calls": tool_calls_parsed}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_resp.get("eval_count", 0),
                    "total_tokens": ollama_resp.get("prompt_eval_count", 0) + ollama_resp.get("eval_count", 0)}
            }

    if tool_calls:
        oai_tool_calls = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            oai_tool_calls.append({"id": tc.get("id", f"call_{i}"), "type": "function",
                "function": {"name": fn.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else args}})
        log(f"🔧 Tool call: {[tc['function']['name'] for tc in oai_tool_calls]}")
        return {
            "id": "chatcmpl-local", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                "tool_calls": oai_tool_calls}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                "completion_tokens": ollama_resp.get("eval_count", 0),
                "total_tokens": ollama_resp.get("prompt_eval_count", 0) + ollama_resp.get("eval_count", 0)}
        }

    log(f"💬 Text response: {content[:120]}...")
    return {
        "id": "chatcmpl-local", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"}],
        "usage": {"prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
            "completion_tokens": ollama_resp.get("eval_count", 0),
            "total_tokens": ollama_resp.get("prompt_eval_count", 0) + ollama_resp.get("eval_count", 0)}
    }


async def ollama_stream_to_oai(ollama_stream, model):
    full_content = ""
    collected_tool_calls = {}
    tool_call_logged = False

    async for line in ollama_stream.aiter_lines():
        if not line.strip(): continue
        try:
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                if not tool_call_logged:
                    log(f"🔧 Stream native tool calls: {[tc.get('function',{}).get('name') for tc in tool_calls]}")
                    tool_call_logged = True
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if i not in collected_tool_calls:
                        collected_tool_calls[i] = {"id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {"name": fn.get("name", ""),
                                "arguments": args if isinstance(args, dict) else {}}}
                    else:
                        if isinstance(args, dict):
                            collected_tool_calls[i]["function"]["arguments"].update(args)

            elif content:
                full_content += content
                yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'content':content},'finish_reason':None}]})}\n\n"

            if chunk.get("done"):
                if collected_tool_calls:
                    for i, tc in collected_tool_calls.items():
                        args_str = json.dumps(tc["function"]["arguments"])
                        delta = {"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": args_str}}]}
                        yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':delta,'finish_reason':None}]})}\n\n"
                    yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                if full_content:
                    xml_calls = parse_xml_tool_calls(full_content)
                    if xml_calls:
                        log(f"🔧 Stream XML tool calls detected, converting...")
                        for i, tc in enumerate(xml_calls):
                            delta = {"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                "function": tc["function"]}]}
                            yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':delta,'finish_reason':None}]})}\n\n"
                        yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                break

        except Exception as e:
            log(f"⚠️ Stream parse error: {e}", "WARN")
            continue


@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_ALIAS, "tunnel": TUNNEL_URL, "kaggle_ip": kaggle_ip}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ALIAS, "object": "model"}]}

@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    rows = ""
    for e in reversed(LOG_BUFFER):
        color = {"INFO": "#00ff88", "WARN": "#ffaa00", "ERROR": "#ff4444",
            "REQ": "#00aaff", "RES": "#aa88ff"}.get(e["level"], "#ffffff")
        msg = e["msg"].replace("<", "&lt;").replace(">", "&gt;")
        rows += f'<tr><td style="color:#888;white-space:nowrap">{e["ts"]}</td><td style="color:{color};padding-left:12px">[{e["level"]}]</td><td style="padding-left:12px;word-break:break-all">{msg}</td></tr>'
    return f"""<!DOCTYPE html>
<html><head><title>Vibe Engine Logs</title>
<meta http-equiv="refresh" content="3">
<style>
  body {{ background: #0d0d0d; color: #eee; font-family: monospace; font-size: 13px; padding: 16px; }}
  h2 {{ color: #00ff88; }} table {{ width: 100%; border-collapse: collapse; }}
  tr:hover {{ background: #1a1a1a; }}
  td {{ padding: 3px 6px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  .info {{ color: #888; font-size: 11px; margin-bottom: 12px; }}
</style></head><body>
<h2>⚡ Vibe Engine Logs</h2>
<div class="info">Model: {MODEL_ALIAS} &nbsp;|&nbsp; IP: {kaggle_ip} &nbsp;|&nbsp; Auto-refresh: 3s &nbsp;|&nbsp; Showing last {len(LOG_BUFFER)} entries</div>
<table>{rows}</table></body></html>"""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global LAST_ACTIVITY
    LAST_ACTIVITY = time.time()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad request"}, status_code=400)

    tools_in = body.get("tools")
    stream = body.get("stream", False)
    msg_count = len(body.get("messages", []))
    log(f"📨 Request — stream={stream}, tools={len(tools_in) if tools_in else 0}, msgs={msg_count}", "REQ")

    # Title generation uses minimal prompt — full prompt causes timeouts
    is_title_request = not tools_in and msg_count <= 3
    messages = [m for m in body.get("messages", []) if m.get("role") != "system"]
    if is_title_request:
        messages = [{"role": "system", "content": "Generate a short 4-6 word title for this conversation."}] + messages
    else:
        messages = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}] + messages
    messages = translate_messages_to_ollama(messages)
    tools = oai_tools_to_ollama(tools_in)
    
    if tools:
        log(f"🛠️  Tools passed to model: {[t['function']['name'] for t in tools]}", "REQ")

    original_tokens = len(json.dumps(body)) // 4
    slim_tokens = len(json.dumps({"messages": messages, "tools": tools or []})) // 4
    log(f"📊 Tokens: original ~{original_tokens} → slim ~{slim_tokens} (saved ~{original_tokens - slim_tokens})", "REQ")

    ollama_body = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "stream": stream,
        "keep_alive": -1,
        "options": {"temperature": 0.15, "num_predict": min(body.get("max_tokens", 4096), 4096)}
    }
    if tools:
        ollama_body["tools"] = tools

    if stream:
        current_model = MODEL_ALIAS
        async def generate():
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream("POST", "http://localhost:8000/api/chat", json=ollama_body) as r:
                    if r.status_code != 200:
                        body_bytes = await r.aread()
                        err = body_bytes.decode()
                        log(f"❌ Ollama rejected: {err[:200]}", "ERROR")
                        if "does not support tools" in err:
                            log("🔄 Retrying without tools (registry bypass)...", "WARN")
                            no_tools_body = {k: v for k, v in ollama_body.items() if k != "tools"}
                            async with client.stream("POST", "http://localhost:8000/api/chat", json=no_tools_body) as r2:
                                async for chunk in ollama_stream_to_oai(r2, current_model):
                                    yield chunk
                            return
                        yield f"data: {json.dumps({'id':'chatcmpl-local','object':'chat.completion.chunk','model':current_model,'choices':[{'index':0,'delta':{'content':'Error: Ollama rejected request'},'finish_reason':'stop'}]})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for chunk in ollama_stream_to_oai(r, current_model):
                        yield chunk
        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post("http://localhost:8000/api/chat", json=ollama_body)
        if r.status_code != 200:
            err = r.text
            log(f"❌ Ollama rejected (non-stream): {err[:200]}", "ERROR")
            if "does not support tools" in err:
                log("🔄 Retrying without tools (registry bypass)...", "WARN")
                no_tools_body = {k: v for k, v in ollama_body.items() if k != "tools"}
                r = await client.post("http://localhost:8000/api/chat", json=no_tools_body)
    try:
        resp = ollama_to_oai_response(r.json(), MODEL_ALIAS)
        log(f"📤 Response — finish_reason={resp['choices'][0]['finish_reason']}", "RES")
        return JSONResponse(resp)
    except Exception as e:
        log(f"❌ Response error: {e}", "ERROR")
        return JSONResponse({"error": str(e), "raw": r.text[:500]}, status_code=500)


log("✅ Middleware ready. Starting server...")
config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="error")
server = uvicorn.Server(config)
await server.serve()
