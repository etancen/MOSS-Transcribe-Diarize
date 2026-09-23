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

    解析器（``TranscriptStreamParser``）会丢弃它无法解析的部分，所以偏离约定格式的
    输出通常只损失对应片段、不会让整次推理作废。

    两个必须知道的例外，都源于解析器的 ``_after_end``：``[end]`` 之后若跟着非空白的
    杂散文本，它会把 ``[end]`` 折回正文并退回"读取正文"状态。

    1. 流尾的杂散文本：该片段永不闭合，``close()`` 也不吐出它——窗口的最后一段丢失。
    2. 中间位置的杂散文本：紧随其后的 ``[时间戳]`` 会被当成**本片段**的结束时间，
       于是本片段文字被污染、结束时间被顶到下一段的起点，而那个 ``[Sxx]`` 在
       ``_READ_START`` 状态下解析失败被丢弃，下一段随之静默消失。

    第 2 种情况下本模块用"正文含方括号即判为污染、整段丢弃"来兜底（见下），把静默
    的文字污染换成干净的丢失。两种丢失都由下一个窗口重新覆盖该音频来恢复。

    另：时间戳顺序颠倒时解析器同样不闭合该片段（``_read_end`` 只接受
    ``end >= start``），且 ``_parse_timestamp`` 不产生负值，所以这里不需要交换分支
    ——那会是一段永远执行不到的死代码。
    """
    parser = TranscriptStreamParser()
    local = list(parser.feed(raw_text))
    local.extend(parser.close())
    out: list[Segment] = []
    for item in local:
        # 在模型的紧凑格式里方括号是结构性的。正文中出现 ``[`` / ``]`` 说明解析器
        # 把某个时间戳折回了正文，即该段已被污染：文字和结束时间都不对。丢弃它
        # 可以把一次静默的定稿污染换成一次干净的丢失，而下一个窗口会重新覆盖这段
        # 音频，所以丢失是暂时性的。没有这个兜底，被污染的段落会被写进定稿且永不修正。
        if "[" in item.text or "]" in item.text:
            continue
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
