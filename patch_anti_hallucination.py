#!/usr/bin/env python3
"""
Anti-hallucination patch for VibeVoice-ASR inference script.
在AutoDL上运行: python /root/autodl-tmp/patch_anti_hallucination.py

设计思路：
  直播音频结构: [话术~10min] → [音乐BGM~1-3min] → [话术~10min] → [音乐] → ...
  主播每轮会重复类似的销售话术 → 这是合法内容，必须正常转写
  音乐段导致模型幻觉（同一短语token级重复几千次）→ 这是噪音，要跳过

  合法重复 vs 幻觉重复的区别:
    - 合法: 主播说"百分百纯羊绒"出现10次，但分散在不同句子里，每次上下文不同
    - 幻觉: "这个是羊绒" 连续重复5000次，没有任何其他内容穿插

三层防幻觉机制:
  1. RepetitionStopCriteria — 检测token级连续重复，立即停止生成（不影响合法话术）
  2. trim_repetitive_tail() — 后处理裁剪被停止前已生成的重复尾巴
  3. repetition_penalty=1.1 — 轻微抑制token重复（设低一点，不影响正常转写质量）

注意: repetition_penalty 不宜设太高（>1.3会影响正常转写准确率），
      主要靠 StoppingCriteria 做硬刹车。
"""

import re
import shutil
from pathlib import Path

SCRIPT_PATH = Path("/root/autodl-tmp/VibeVoice/demo/vibevoice_asr_inference_from_file.py")

# =====================================================
# 要注入到推理脚本的代码
# =====================================================

ANTI_HALLUCINATION_CODE = '''
# ====================================================================
# Anti-hallucination module (auto-patched)
# 检测音乐段导致的模型幻觉，提前停止生成，保留有效转写内容
# ====================================================================
from transformers import StoppingCriteria, StoppingCriteriaList

class RepetitionStopCriteria(StoppingCriteria):
    """
    检测模型生成中的token级连续重复（音乐幻觉特征），提前停止。

    判断逻辑:
      每生成80个token检查一次，取最后600个token解码为文本，
      如果末尾N个字符在这段文本中出现了>=threshold次，判定为幻觉。

    不会误杀合法重复话术:
      主播重复说"百分百纯羊绒"，但每次都嵌在不同的完整句子里，
      600个token（约200-300个汉字）的窗口内，同一个30字短语不会出现4次。
      而音乐幻觉是"你高你高你高..."连续几百次，轻松触发阈值。
    """
    def __init__(self, tokenizer, check_every=80, ngram_chars=20, threshold=4):
        self.tokenizer = tokenizer
        self.check_every = check_every
        self.ngram_chars = ngram_chars
        self.threshold = threshold
        self.call_count = 0
        self.stopped = False
        self.valid_token_count = 0  # 记录停止前的有效token数

    def __call__(self, input_ids, scores, **kwargs):
        self.call_count += 1
        if self.call_count % self.check_every != 0:
            return False

        # 取最后600个token解码
        n_tokens = input_ids.shape[1]
        tail_start = max(0, n_tokens - 600)
        text = self.tokenizer.decode(input_ids[0, tail_start:], skip_special_tokens=True)

        if len(text) < self.ngram_chars * (self.threshold + 2):
            return False

        # 用多个窗口大小检测重复
        for n in [self.ngram_chars, max(8, self.ngram_chars // 2)]:
            tail_phrase = text[-n:]
            if not tail_phrase.strip():
                continue
            count = text.count(tail_phrase)
            if count >= self.threshold:
                self.stopped = True
                self.valid_token_count = max(0, n_tokens - 600)  # 保守估计有效部分
                print(f"    [AntiHallucination] MUSIC DETECTED - token级重复在第{self.call_count * self.check_every}个token触发")
                print(f"    重复短语({count}次): '{tail_phrase[:40]}...'")
                print(f"    已停止生成，保留前部有效内容")
                return True
        return False


def trim_repetitive_tail(text, min_phrase_len=10, max_repeats=3):
    """
    后处理：裁剪生成文本末尾的重复内容。

    即使StoppingCriteria触发了，在触发前可能已经生成了一些重复内容。
    这个函数找到重复开始的位置，保留第一次出现，删除后续重复。

    不会影响合法内容:
      只检测"同一短语连续出现N次"的模式（如"这个是羊绒这个是羊绒这个是羊绒..."）
      主播话术中的类似句子不会是完全相同的短语首尾相连。
    """
    if not text or len(text) < min_phrase_len * max_repeats * 2:
        return text

    best_cut = len(text)

    # 从短到长尝试不同的phrase长度
    for phrase_len in range(min_phrase_len, min(100, len(text) // 4)):
        tail = text[-phrase_len:]
        if not tail.strip():
            continue
        count = text.count(tail)
        if count >= max_repeats:
            # 找到重复区域的起点
            first = text.find(tail)
            second = text.find(tail, first + 1)
            if second > 0 and second < best_cut:
                best_cut = second
            break

    if best_cut < len(text):
        trimmed = len(text) - best_cut
        print(f"    [TrimTail] 裁剪了{trimmed}个字符的重复尾巴")
        return text[:best_cut]

    return text
# ==================================================================
'''


def patch():
    if not SCRIPT_PATH.exists():
        print(f"ERROR: {SCRIPT_PATH} not found")
        return False

    # Backup: always keep the ORIGINAL (pre-any-patch) version
    backup = SCRIPT_PATH.with_suffix('.py.bak_original')
    if not backup.exists():
        shutil.copy2(SCRIPT_PATH, backup)
        print(f"Original backup saved to {backup}")
    else:
        print(f"Original backup exists at {backup}")

    # If there's a previous patch, start fresh from original
    if SCRIPT_PATH.with_suffix('.py.bak').exists():
        # There was a previous patch attempt, restore from original
        shutil.copy2(backup, SCRIPT_PATH)
        print("Restored from original backup to apply clean patch.")

    # Also save a pre-this-patch backup
    shutil.copy2(SCRIPT_PATH, SCRIPT_PATH.with_suffix('.py.bak'))

    code = SCRIPT_PATH.read_text(encoding='utf-8')
    patches_applied = 0

    # =========================================================
    # PATCH 1: Inject anti-hallucination module
    # =========================================================
    if 'RepetitionStopCriteria' not in code:
        lines = code.split('\n')
        # Find the last top-level import
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
            if i > 60:
                break
        lines.insert(insert_idx, ANTI_HALLUCINATION_CODE)
        code = '\n'.join(lines)
        patches_applied += 1
        print("✓ PATCH 1: Injected RepetitionStopCriteria + trim_repetitive_tail")
    else:
        print("- PATCH 1: Already present, skip")

    # =========================================================
    # PATCH 2: Add repetition_penalty=1.1 to model.generate()
    #   (轻微值，不影响正常转写质量)
    # =========================================================
    if 'repetition_penalty' not in code:
        pattern = r'(output_ids\s*=\s*self\.model\.generate\()'
        if re.search(pattern, code):
            code = re.sub(pattern, r'\1\n                repetition_penalty=1.1,', code)
            patches_applied += 1
            print("✓ PATCH 2: Added repetition_penalty=1.1")
        else:
            print("✗ PATCH 2: Could not find model.generate() - needs manual edit")
    else:
        print("- PATCH 2: Already present, skip")

    # =========================================================
    # PATCH 3: Add StoppingCriteria to model.generate()
    # =========================================================
    if 'stopping_criteria' not in code:
        gen_pattern = r'(\s+)(output_ids\s*=\s*self\.model\.generate\()'
        match = re.search(gen_pattern, code)
        if match:
            indent = match.group(1)
            setup = (
                f'\n{indent}# 音乐幻觉检测器: 发现token级重复立即刹车\n'
                f'{indent}_halt = RepetitionStopCriteria(self.tokenizer, check_every=80, ngram_chars=20, threshold=4)\n'
                f'{indent}_halt_list = StoppingCriteriaList([_halt])\n'
            )
            code = code[:match.start()] + setup + code[match.start():]

            # Add stopping_criteria param to generate()
            code = re.sub(
                r'(output_ids\s*=\s*self\.model\.generate\()',
                r'\1\n                stopping_criteria=_halt_list,',
                code,
                count=1
            )
            patches_applied += 1
            print("✓ PATCH 3: Added stopping_criteria to generate()")
        else:
            print("✗ PATCH 3: Could not find generate() pattern - needs manual edit")
    else:
        print("- PATCH 3: Already present, skip")

    # =========================================================
    # PATCH 4: Add trim_repetitive_tail() after decode
    # =========================================================
    if 'trim_repetitive_tail' not in code or 'trim_repetitive_tail(raw' not in code:
        # Find tokenizer.decode calls and add trim after them
        decode_pattern = r'(raw_text\s*=\s*self\.tokenizer\.decode\([^)]+\))'
        matches = list(re.finditer(decode_pattern, code))
        if matches:
            # Replace from end to start to preserve positions
            for m in reversed(matches):
                insert_pos = m.end()
                code = code[:insert_pos] + '\n            raw_text = trim_repetitive_tail(raw_text)' + code[insert_pos:]
            patches_applied += 1
            print(f"✓ PATCH 4: Added trim_repetitive_tail() after {len(matches)} decode() calls")
        else:
            print("✗ PATCH 4: Could not find decode pattern - trim not auto-applied")
            print("  (StoppingCriteria will still catch hallucination during generation)")
    else:
        print("- PATCH 4: Already present, skip")

    # =========================================================
    # Write result
    # =========================================================
    SCRIPT_PATH.write_text(code, encoding='utf-8')
    print(f"\n{'='*60}")
    print(f"Applied {patches_applied} patches to {SCRIPT_PATH}")
    print(f"Original preserved at {backup}")
    print(f"{'='*60}")

    if patches_applied > 0:
        print("""
使用方法（和之前完全一样）:

  # 直播模式（10分钟切片 + 幻觉检测）
  bash /root/autodl-tmp/run_asr.sh live "/root/autodl-tmp/2月7号20点2h12.MP3"

  # 课程模式（50分钟切片，短音频不切）
  bash /root/autodl-tmp/run_asr.sh course /root/autodl-tmp/主播课程音频

运行时会看到类似日志:
  [AntiHallucination] MUSIC DETECTED - token级重复在第2400个token触发
  重复短语(15次): '这个是羊绒，这个是羊绒，这个是羊绒...'
  已停止生成，保留前部有效内容

如需恢复原始脚本:
  cp /root/autodl-tmp/VibeVoice/demo/vibevoice_asr_inference_from_file.py.bak_original \\
     /root/autodl-tmp/VibeVoice/demo/vibevoice_asr_inference_from_file.py
""")

    return True


if __name__ == '__main__':
    patch()
