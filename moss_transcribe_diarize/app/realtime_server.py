"""实时转写的本地服务：WebSocket 收音频，HTTP 看历史与导出。

**本模块不得 import torch。** ``--backend vllm`` 下它必须能在一个没装 torch 的进程里
跑起来——这正是阶段一第一件事做的那层隔离的用途。转写器由 ``transcriber_factory``
注入，本模块不知道它是 vLLM 还是本地模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.export import EXPORT_FORMATS, export_session
from moss_transcribe_diarize.realtime.store import SessionStore

STATIC_DIR = Path(__file__).with_name("static")


def _speaker_names(meta: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(item.get("name") or item["id"])
        for item in meta.get("speakers") or []
        if isinstance(item, dict) and item.get("id")
    }


def _with_hotwords(prompt: str, hotwords: Any) -> str:
    """把热词拼进 prompt——模型只认 prompt，没有单独的热词参数。

    ``_start`` 与 ``set_hotwords`` 都走这里，免得两处各写一份拼接逻辑。
    """
    words = [str(word).strip() for word in (hotwords or []) if str(word).strip()]
    if not words:
        return prompt.strip()
    return "{} 热词提示：{}".format(prompt.strip(), ", ".join(words)).strip()


class SessionRegistry:
    """活跃会话的登记处。每个 WebSocket 连接一个会话，互不共享状态。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def add(self, session_id: str, session: Any) -> None:
        self._sessions[session_id] = session

    def get(self, session_id: str) -> Any | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active(self) -> list[str]:
        return sorted(self._sessions)


def create_realtime_app(
    *,
    config: RealtimeConfig,
    transcriber_factory: Callable[[], Any],
    embedder: Any = None,
    runs_dir: str | Path = "runs/realtime",
    static_dir: str | Path | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to run the realtime service.") from exc

    app = FastAPI(title="MOSS Realtime")
    runs = Path(runs_dir).expanduser()
    runs.mkdir(parents=True, exist_ok=True)
    assets = Path(static_dir).expanduser() if static_dir else STATIC_DIR
    registry = SessionRegistry()
    app.state.registry = registry
    app.state.config = config
    app.state.runs_dir = runs

    def error(code: str, detail: str, status: int = 400):
        return JSONResponse({"detail": detail, "code": code}, status_code=status)

    def load_rows(session_id: str) -> list[dict]:
        try:
            return SessionStore.load_committed(runs, session_id)
        except ValueError as exc:                      # session_id 非法（含路径穿越）
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def load_meta(session_id: str) -> dict:
        try:
            return SessionStore.load_meta(runs, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = assets / "realtime.html"
        if not page.exists():
            return HTMLResponse("<h1>MOSS Realtime</h1><p>realtime.html 尚未提供。</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime")
    def runtime():
        return {
            "config": {
                "window": config.window,
                "hop": config.hop,
                "tail": config.tail,
                "silence_gate": config.silence_gate,
                "speaker_threshold": config.speaker_threshold,
                "poll_interval": config.poll_interval,
            },
            "backend": _transcriber_info(transcriber_factory),
            "speaker": {
                "enabled": embedder is not None,
                "provider": type(embedder).__name__ if embedder is not None else None,
            },
            "active_sessions": registry.active(),
            "export_formats": list(EXPORT_FORMATS),
        }

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": SessionStore.list_sessions(runs)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        return {"session": meta, "segments": load_rows(session_id)}

    @app.get("/api/sessions/{session_id}/audio")
    def get_audio(session_id: str):
        path = SessionStore.session_dir(runs, session_id) / "audio.wav"
        if not path.exists():
            return error("audio_missing", "this session has no recording", 404)
        return FileResponse(path, filename=path.name)

    @app.get("/api/sessions/{session_id}/export")
    def export(session_id: str, format: str = Query("srt")):
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        rows = load_rows(session_id)
        try:
            text = export_session(rows, format, speaker_names=_speaker_names(meta))
        except ValueError as exc:
            return error("invalid_format", str(exc), 400)
        return JSONResponse({"format": format, "text": text})

    return app


def _transcriber_info(transcriber_factory: Callable[[], Any]) -> dict[str, Any]:
    """尽量在不构造转写器的前提下描述后端。"""
    return {"factory": getattr(transcriber_factory, "__name__", repr(transcriber_factory))}
