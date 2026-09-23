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

    解析器（``TranscriptStreamParser``）会丢弃它无法解析的部分，所以多数格式偏离
    只损失对应片段、不会让整次推理作废。

    两个必须知道的例外，都源于解析器的 ``_after_end``：``[end]`` 之后若跟着非空白的
    杂散文本，它会把 ``[end]`` 折回正文并退回"读取正文"状态。

    1. 流尾的杂散文本：该片段永不闭合，``close()`` 也不吐出它——窗口的最后一段丢失。
    2. 中间位置的杂散文本：紧随其后的 ``[时间戳]`` 会被当成**本片段**的结束时间，
       于是本片段文字被污染（正文里混入 ``[1.5]垃圾`` 之类）、结束时间被顶到下一段的
       起点，而那个 ``[Sxx]`` 在 ``_READ_START`` 状态下解析失败被丢弃，下一段随之
       静默消失。

    本模块**不**去修补这两种情况，因为修补不了：解析器把"正文里本来就有的方括号"
    和"被折回的时间戳"处理成同一种结果（都走 ``_after_end``，后面跟的都是任意文本），
    所以任何"正文含方括号就丢弃"之类的判据都会误伤合法内容——正文里的方括号是
    ``transcript_parser`` 有测试固定下来的刻意保留行为。误伤的代价是静默丢内容，比
    被污染的段落更隐蔽。

    两种例外都由**下一个窗口重新覆盖该段音频**来恢复，属于暂时性丢失而非永久丢失。

    另：时间戳顺序颠倒时解析器同样不闭合该片段（``_read_end`` 只接受
    ``end >= start``），且 ``_parse_timestamp`` 不产生负值，所以这里不需要交换分支
    ——那会是一段永远执行不到的死代码。
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
