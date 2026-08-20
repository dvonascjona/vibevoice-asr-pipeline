#!/usr/bin/env python3
"""
合并说话人标签到 SRT
用法: python merge_speaker.py 转写.srt 分割.json [输出.srt]
"""
import os, sys, json, re, argparse

def parse_srt_time(t):
    t = t.replace(",", ".")
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def fmt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("srt", help="ASR 转写 SRT 文件")
    parser.add_argument("diar_json", help="diarize_only.py 输出的 JSON")
    parser.add_argument("output", nargs="?", help="输出 SRT（默认覆盖原文件旁边加 _meeting 后缀）")
    args = parser.parse_args()

    # 读取分割结果
    with open(args.diar_json, encoding="utf-8") as f:
        diar = json.load(f)
    diar_segs = diar["segments"]  # [{start, end, speaker}]
    print(f"[Merge] 分割结果: {len(diar_segs)} 段, {len(diar['speakers'])} 位说话人")

    # 读取 SRT
    with open(args.srt, encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    parsed = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        idx = lines[0]
        time_match = re.match(r'(.+?) --> (.+)', lines[1])
        if not time_match:
            continue
        start = parse_srt_time(time_match.group(1))
        end = parse_srt_time(time_match.group(2))
        text = " ".join(lines[2:])
        parsed.append({"idx": idx, "start": start, "end": end, "text": text})

    print(f"[Merge] SRT: {len(parsed)} 条字幕")

    # 为每条字幕找说话人
    def find_speaker(s_start, s_end):
        dur = max(s_end - s_start, 0.001)
        overlap = {}
        for d in diar_segs:
            ov = max(0, min(s_end, d["end"]) - max(s_start, d["start"]))
            if ov > 0:
                sp = d["speaker"]
                overlap[sp] = overlap.get(sp, 0) + ov
        if overlap:
            return max(overlap, key=overlap.get)
        # 找最近的
        min_gap, nearest = float("inf"), None
        for d in diar_segs:
            gap = max(s_start - d["end"], d["start"] - s_end, 0)
            if gap < min_gap:
                min_gap, nearest = gap, d["speaker"]
        return nearest if nearest and min_gap < 3.0 else None

    # 写输出 SRT
    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.srt)
        out_path = base + "_meeting" + ext

    # 写带标签 SRT
    spk_stats = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in parsed:
            spk = find_speaker(seg["start"], seg["end"])
            prefix = f"[{spk}] " if spk else ""
            f.write(f"{seg['idx']}\n")
            f.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n")
            f.write(f"{prefix}{seg['text']}\n\n")
            if spk:
                spk_stats[spk] = spk_stats.get(spk, 0) + (seg["end"] - seg["start"])

    print(f"[Merge] 输出: {out_path}")

    # 写 TXT 发言汇总
    txt_path = out_path.replace(".srt", "_发言汇总.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n会议转写 — 发言人区分版\n" + "=" * 60 + "\n\n")
        prev_spk, buffer, seg_start = None, [], 0
        def flush(spk, buf, t):
            if not buf: return
            m, s = int(t//60), int(t%60)
            f.write(f"[{spk}] {m:02d}:{s:02d}\n")
            f.write("".join(buf).strip() + "\n\n")
        for seg in parsed:
            spk = find_speaker(seg["start"], seg["end"]) or "Unknown"
            if spk != prev_spk:
                flush(prev_spk, buffer, seg_start)
                buffer, prev_spk, seg_start = [seg["text"] + " "], spk, seg["start"]
            else:
                buffer.append(seg["text"] + " ")
        flush(prev_spk, buffer, seg_start)
        f.write("=" * 60 + "\n发言统计\n" + "=" * 60 + "\n")
        total = sum(spk_stats.values())
        for spk in sorted(spk_stats):
            dur = spk_stats[spk]
            f.write(f"  {spk}: {dur/60:.1f}分钟 ({dur/total*100:.0f}%)\n")

    print(f"[Merge] 发言汇总: {txt_path}")
    print("\n完成！")

if __name__ == "__main__":
    main()
