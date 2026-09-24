"""``mtd-realtime`` 的入口。

**本模块不得 import torch。** ``--backend hf`` 的实现里才会去 import
``ModelRunner``（那是本地跑模型必须的），``--backend vllm`` 下永远不加载——所以那句
import 写在工厂函数里面，不写在文件顶部。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from moss_transcribe_diarize.realtime.config import RealtimeConfig

LOOPBACK_WARNING = (
    "服务默认只绑 127.0.0.1，且**没有鉴权**。绑到 0.0.0.0 等于把会议录音的转写接口"
    "开放给同网段，请只在可信网络里这么做。"
)


def _make_output_robust() -> None:
    """这个 CLI 的帮助与提示都是中文，而 Windows 上的默认编码跟着区域设置走。

    英文 Windows 是 cp1252，`mtd-realtime --help | more` 会直接抛 UnicodeEncodeError。
    降级成替换字符：文字可能缺几个字，但程序不能因为一句提示就崩。终端本身是中文编码
    时（cp936）不受影响。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:                                   # 被换成了非文本流
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):                             # 流已关闭/不可重配
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mtd-realtime",
        description="实时会议语音转文字。默认绑 127.0.0.1:7870；服务无鉴权，别直接暴露到公网。",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize",
                        help="--backend hf 时的模型路径或 HF id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vllm-api-key", default=None)
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--hop", type=float, default=5.0)
    parser.add_argument("--tail", type=float, default=6.0)
    parser.add_argument("--silence-gate", dest="silence_gate", action="store_true", default=True)
    parser.add_argument("--no-silence-gate", dest="silence_gate", action="store_false")
    parser.add_argument("--record", dest="record", action="store_true", default=True)
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="不落盘完整录音，只保留转写文本")
    parser.add_argument("--speaker", dest="speaker", action="store_true", default=True)
    parser.add_argument("--no-speaker", dest="speaker", action="store_false",
                        help="关闭全局说话人一致，退化为每窗口局部标签")
    parser.add_argument("--speaker-model", default=None, help="本地声纹模型路径；默认自动下载")
    parser.add_argument("--speaker-threshold", type=float, default=0.55)
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="声纹模型语言")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="输出 token 上限；默认按每次窗口的实际长度推算")
    parser.add_argument("--runs-dir", default="runs/realtime")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RealtimeConfig:
    return RealtimeConfig(
        window=args.window,
        hop=args.hop,
        tail=args.tail,
        silence_gate=args.silence_gate,
        speaker_threshold=args.speaker_threshold,
        max_new_tokens=args.max_new_tokens,
        poll_interval=args.poll_interval,
    )


def build_embedder(args: argparse.Namespace) -> Any:
    if not args.speaker:
        return None
    from moss_transcribe_diarize.realtime.speaker import OnnxCampplusEmbedder

    return OnnxCampplusEmbedder(model_path=args.speaker_model, lang=args.lang)


def build_transcriber_factory(args: argparse.Namespace, config: RealtimeConfig) -> Callable[[], Any]:
    """每次新会话调一次——`HfWindowTranscriber` 只能串行使用，每会话必须独立实例。"""
    if args.backend == "vllm":
        if not args.vllm_base_url:
            raise SystemExit("--backend vllm 需要 --vllm-base-url（例如 http://127.0.0.1:8000）")
        from moss_transcribe_diarize.realtime.transcriber import VllmWindowTranscriber

        def make_vllm() -> Any:
            return VllmWindowTranscriber(
                base_url=args.vllm_base_url,
                model=args.vllm_model or args.model,
                api_key=args.vllm_api_key,
                token_budget_floor=config.effective_max_new_tokens(),
                max_new_tokens=args.max_new_tokens,
            )

        return make_vllm

    def make_hf() -> Any:
        import tempfile

        from moss_transcribe_diarize.app.model_runner import ModelRunner      # 只有这条路需要 torch
        from moss_transcribe_diarize.realtime.transcriber import HfWindowTranscriber

        runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
        scratch = Path(tempfile.mkdtemp(prefix="mtd-realtime-"))
        return HfWindowTranscriber(
            runner, scratch,
            token_budget_floor=config.effective_max_new_tokens(),
            max_new_tokens=args.max_new_tokens,
        )

    return make_hf


def build_probe(args: argparse.Namespace) -> Callable[[], dict] | None:
    """``/api/runtime`` 的连通性探测；只有远端后端才有可探的东西。"""
    if args.backend != "vllm" or not args.vllm_base_url:
        return None
    from moss_transcribe_diarize.app.openai_audio_client import check_endpoint

    def probe() -> dict:
        result = check_endpoint(args.vllm_base_url, api_key=args.vllm_api_key or "EMPTY")
        return {"kind": "vllm", "base_url": args.vllm_base_url, "model": args.vllm_model or args.model, **result}

    return probe


def main(argv: list[str] | None = None) -> int:
    _make_output_robust()
    args = parse_args(argv)
    config = build_config(args)
    embedder = build_embedder(args)
    factory = build_transcriber_factory(args, config)

    from moss_transcribe_diarize.app.realtime_server import create_realtime_app

    app = create_realtime_app(
        config=config,
        transcriber_factory=factory,
        embedder=embedder,
        runs_dir=args.runs_dir,
        probe=build_probe(args),
        record_audio=args.record,
    )

    print(LOOPBACK_WARNING)
    print(f"后端 {args.backend}；打开 http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
