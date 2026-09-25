/**
 * 实时页面纯函数的测试。跑法：`node --test tests/js/`（pytest 里有一条包装用例会跑它）。
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  applyRename,
  formatClock,
  mergeCommitted,
  pipelineHint,
  segmentsForSpeaker,
  shouldFollow,
  speakerColor,
} from "../../moss_transcribe_diarize/app/static/realtime_logic.js";

test("formatClock 按需要决定要不要带小时", () => {
  assert.equal(formatClock(0), "00:00");
  assert.equal(formatClock(65), "01:05");
  assert.equal(formatClock(3725), "1:02:05");
  assert.equal(formatClock(-4), "00:00");
  assert.equal(formatClock(undefined), "00:00");
});

test("说话人颜色只由 id 决定，同一个 id 每次同色", () => {
  assert.equal(speakerColor("S01"), speakerColor("S01"));
  assert.notEqual(speakerColor("S01"), speakerColor("S02"));
  assert.equal(speakerColor("S09"), speakerColor("S01"));  // 调色板有 8 个槽位，回绕
});

test("committed 是追加语义，但同一个 id 再来一次是原地替换", () => {
  const first = [{ id: "seg-1", speaker: "S01" }, { id: "seg-2", speaker: "S01" }];

  const merged = mergeCommitted(first, [{ id: "seg-1", speaker: "S02" }, { id: "seg-3", speaker: "S02" }]);

  assert.equal(merged.length, 3);
  assert.equal(merged[0].speaker, "S02");
  assert.equal(merged[2].id, "seg-3");
});

test("改名要作用在已经显示过的段落上", () => {
  const segments = [{ id: "seg-1", speaker: "S01", speaker_name: "S01" }, { id: "seg-2", speaker: "S02" }];

  const renamed = applyRename(segments, "S01", "张总");

  assert.equal(renamed[0].speaker_name, "张总");
  assert.equal(renamed[1].speaker_name, undefined);
  assert.equal(segments[0].speaker_name, "S01", "不该改到原数组");
});

test("segmentsForSpeaker 只挑同一个说话人的段", () => {
  const segments = [
    { id: "seg-1", speaker: "S01" },
    { id: "seg-2", speaker: "S01" },
    { id: "seg-3", speaker: "S02" },
  ];

  assert.deepEqual(segmentsForSpeaker(segments, "S01"), ["seg-1", "seg-2"]);
  assert.deepEqual(segmentsForSpeaker(segments, "S99"), []);
});

test("shouldFollow 看的是距底部多少像素，不是 scrollTop", () => {
  const container = { scrollHeight: 1000, scrollTop: 400, clientHeight: 500 };

  assert.equal(shouldFollow(container), false);                          // 距底 100
  assert.equal(shouldFollow({ ...container, scrollTop: 480 }), true);    // 距底 20
  assert.equal(shouldFollow(null), true);
});

test("有内容可看时不解释", () => {
  assert.equal(pipelineHint({ sending: true, committed: 1 }), null);
  assert.equal(pipelineHint({ sending: true, provisional: 2 }), null);
});

test("没开始时说没开始", () => {
  assert.equal(pipelineHint({ sending: false }).key, "realtime.hint.idle");
});

test("开了但还没有第一条 status 时说是在攒第一个窗口", () => {
  const hint = pipelineHint({ sending: true, hasStatus: false });

  assert.equal(hint.key, "realtime.hint.warmingUp");
});

test("窗口跑过、没被跳过、也还没内容时说是还在等稳定内容", () => {
  const hint = pipelineHint({ sending: true, hasStatus: true });

  assert.equal(hint.key, "realtime.hint.waiting");
});

test("被门控跳过过又一直没内容时，指向麦克风", () => {
  const hint = pipelineHint({ sending: true, hasStatus: true, gatedWindows: 3 });

  assert.equal(hint.key, "realtime.hint.silentWindows");
  assert.equal(hint.params.count, 3);
});

test("后端在连续失败时，降级优先于静音猜测", () => {
  const hint = pipelineHint({ sending: true, hasStatus: true, gatedWindows: 3, failures: 2 });

  assert.equal(hint.key, "realtime.hint.degraded");
  assert.equal(hint.params.count, 2);

  assert.equal(
    pipelineHint({ sending: true, hasStatus: true, degraded: true }).key,
    "realtime.hint.degraded",
  );
});
