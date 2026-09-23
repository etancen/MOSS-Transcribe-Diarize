"""Turn one window's raw model output into window-normalised segments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from moss_transcribe_diarize.transcript_parser import TranscriptStreamParser

MIN_SEGMENT_DURATION = 0.05


@dataclass(frozen=True, slots=True)
class Segment:
    """一个转写段落。所有时间都是相对会话开始的绝对秒。"""

    start: float
    end: float
    speaker: str
    text: str
    window_id: int


def parse_window_segments(raw_text: str, window_start: float, window_id: int) -> list[Segment]:
    """解析模型输出，把窗口内的局部时间换算成绝对时间。

    解析器（``TranscriptStreamParser``）本身会丢弃无法解析的片段，所以偏离约定
    格式的输出只损失对应片段，不会让整次推理作废；只要后面还有合法的
    ``[start][Sxx]``，后续片段照常产出。

    有一个例外要知道：``[end]`` 之后若跟着非空白的杂散文本，解析器会把 ``[end]``
    折回正文、该片段永不闭合，于是**窗口的最后一段会丢**。这是解析器刻意的宽松
    恢复策略，不在本模块修正范围。代价可接受——下一个窗口会重新覆盖这段音频，
    所以是暂时性丢失而非永久丢失。

    另：时间戳顺序颠倒时解析器同样不闭合该片段（它只接受 ``end >= start``），
    且 ``_parse_timestamp`` 不产生负值，所以这里不需要交换分支。
    """
    parser = TranscriptStreamParser()
    local = list(parser.feed(raw_text))
    local.extend(parser.close())
    out: list[Segment] = []
    for item in local:
        start = window_start + float(item.start)
        end = window_start + float(item.end)
        out.append(
            Segment(
                start=start,
                end=end,
                speaker=item.speaker or "",
                text=item.text,
                window_id=window_id,
            )
        )
    return out


def clamp_to_window(seg: Segment, window_start: float, window_end: float) -> Segment | None:
    """把段落裁剪进窗口，或判定它应当被丢弃。

    跨出**左**边界的段落裁剪起点——模型偶尔会给出早于窗口起点的时间戳，裁剪比
    丢弃少丢内容。跨出**右**边界的段落直接丢弃——右边界就是"现在"，下一轮推理
    会重新看到它，而裁剪会让文字长度和时间跨度对不上。
    """
    if seg.end <= window_start or seg.start >= window_end:
        return None
    if seg.end > window_end:
        return None
    start = max(seg.start, window_start)
    if seg.end - start < MIN_SEGMENT_DURATION:
        return None
    if start == seg.start:
        return seg
    return replace(seg, start=start)
