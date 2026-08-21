#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 真实验收 v2：区分"真丢讲课内容" vs "课间休息/提问间隙"。

关键教训（2026-08-21）：
  旧版用缺口段的 max_volume(峰值) 判断"有没有语音"。但课间休息时学员走动/咳嗽
  会有 -28dB 的零星尖峰，峰值法把休息误判成"内容丢失"。实测对比（15号）：
    休息段 133-649s: 活跃语音占比 0%,  mean -62dB, 峰值 -28.7dB(← 旧法在这里误判)
    讲课段 700-1200s: 活跃语音占比 53%, mean -49dB, 峰值 -20dB
  改用 silencedetect 测"活跃语音占比"：讲课≈50%+，休息≈0%，25% 阈值干净分开。

判定：对每个超阈值的缺口(头/尾/中)，测其活跃语音占比：
  占比 >= SPEECH_RATIO_FAIL  -> 真丢讲课内容 = FAIL
  占比 <  SPEECH_RATIO_FAIL  -> 课间休息/间歇静音 = 放行(SRT本就该跳过这类段)

用法:
  目录模式:   python3 verify.py <srt_dir> <audio_dir1> [audio_dir2 ...]
  单文件模式: python3 verify.py --single <srt_path> <audio_path>   (exit 0=OK, 1=FAIL)
"""
import os, re, sys, glob, json, subprocess

HEAD_GAP = 60.0
TAIL_GAP = 120.0
MID_GAP  = 120.0
SPEECH_RATIO_FAIL = 0.25   # 缺口内活跃语音占比 >=25% 判真丢内容(实测休息0% vs 讲课53%)
NOISE_DB = "-35dB"         # silencedetect 噪声门限
MIN_SIL  = 2               # 连续静音 >=2s 才算一段静音(滤掉句间自然停顿)
MEASURE_CAP = 2400.0       # 单缺口最多实测 40 分钟, 够判断

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

def dur(path):
    r = sh(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path])
    try: return float(r.stdout.strip())
    except: return 0.0

def speech_ratio(path, ss, gap):
    """在 [ss, ss+gap] 内用 silencedetect 测活跃语音占比(0~1)与平均音量dB。
    返回 (ratio, mean_db)；取不到返回 (None, None)。"""
    if gap <= 1.0:
        return None, None
    span = min(gap, MEASURE_CAP)
    r = sh(["ffmpeg","-hide_banner","-nostats","-ss",f"{ss:.2f}","-t",f"{span:.2f}",
            "-i",path,"-af",f"silencedetect=noise={NOISE_DB}:d={MIN_SIL},volumedetect",
            "-f","null","-"])
    out = r.stderr + r.stdout
    sil = sum(float(m) for m in re.findall(r"silence_duration:\s*([\d.]+)", out))
    sil = min(sil, span)
    ratio = max(0.0, 1.0 - sil / span)
    mm = re.search(r"mean_volume:\s*(-?[\d.]+) dB", out)
    mean_db = float(mm.group(1)) if mm else None
    return ratio, mean_db

def parse_srt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        c = f.read()
    ts = re.findall(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", c)
    out = []
    for a,b,cc,d,e,f_,g,h in ts:
        out.append((int(a)*3600+int(b)*60+int(cc)+int(d)/1000.0,
                    int(e)*3600+int(f_)*60+int(g)+int(h)/1000.0))
    return sorted(out)

def fmt(s):
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def check_one(srt_path, audio_path):
    """返回 (ok: bool, issues: list[str], detail: dict)"""
    adur = dur(audio_path) if audio_path else 0.0
    segs = parse_srt(srt_path)
    if not segs:
        return False, ["SRT为空"], dict(audio_sec=adur, srt_end=0, subs=0)

    first, last = segs[0][0], segs[-1][1]
    head, tail = first, max(0.0, adur - last)
    mid, mid_at = 0.0, 0.0
    for i in range(1, len(segs)):
        g = segs[i][0] - segs[i-1][1]
        if g > mid: mid, mid_at = g, segs[i-1][1]

    # (标签, 缺口长度, 缺口起点)
    checks = []
    if audio_path:
        if head > HEAD_GAP: checks.append(("开头", head, 0.0))
        if tail > TAIL_GAP: checks.append(("尾部", tail, last))
        if mid  > MID_GAP:  checks.append(("中间", mid,  mid_at))

    issues, notes = [], []
    for where, gap, start in checks:
        ratio, mean_db = speech_ratio(audio_path, start, gap)
        if ratio is None:
            issues.append(f"{where}缺{fmt(gap)}(未能实测)")
        elif ratio >= SPEECH_RATIO_FAIL:
            issues.append(f"{where}缺{fmt(gap)}真丢内容(语音占比{ratio*100:.0f}%,均{mean_db:.0f}dB)")
        else:
            notes.append(f"{where}{fmt(gap)}休息/间歇(语音占比{ratio*100:.0f}%)")

    detail = dict(audio_sec=round(adur,1), srt_end=round(last,1), subs=len(segs),
                  head=round(head,1), tail=round(tail,1), mid=round(mid,1), notes=notes)
    return (len(issues) == 0), issues, detail

def main_single():
    srt_path, audio_path = sys.argv[2], sys.argv[3]
    ok, issues, detail = check_one(srt_path, audio_path if os.path.isfile(audio_path) else None)
    tail_note = ("  [" + "; ".join(detail.get("notes", [])) + "]") if detail.get("notes") else ""
    if ok:
        print(f"OK 覆盖完整 (字幕{detail['subs']}条, 音频{fmt(detail['audio_sec'])}, 止于{fmt(detail['srt_end'])}){tail_note}")
        return 0
    else:
        print(f"FAIL {' + '.join(issues)}{tail_note}")
        return 1

def main_dir():
    srt_dir = sys.argv[1]
    audio_dirs = sys.argv[2:]
    amap = {}
    for d in audio_dirs:
        for p in glob.glob(os.path.join(d, "*")):
            if os.path.splitext(p)[1].lower() in (".wav",".mp3",".flac",".m4a",".mp4"):
                amap.setdefault(os.path.splitext(os.path.basename(p))[0], p)

    srts = sorted(glob.glob(os.path.join(srt_dir, "*.srt")))
    if not srts:
        print("没有找到 SRT 文件"); return 1

    rows, bad = [], []
    print(f"{'文件':<36}{'音频':>10}{'字幕止':>10}{'条数':>7}{'头缺':>10}{'尾缺':>10}{'空洞':>10}   判定")
    print("=" * 118)

    for sp in srts:
        name = os.path.splitext(os.path.basename(sp))[0]
        ap = amap.get(name)
        ok, issues, detail = check_one(sp, ap)
        verdict = "✅ 通过" if ok else "❌ " + " + ".join(issues)
        if not ok: bad.append((name, " + ".join(issues) if issues else "SRT为空"))
        print(f"{name[:34]:<36}{fmt(detail.get('audio_sec',0)):>10}{fmt(detail.get('srt_end',0)):>10}{detail.get('subs',0):>7}"
              f"{fmt(detail.get('head',0)):>10}{fmt(detail.get('tail',0)):>10}{fmt(detail.get('mid',0)):>10}   {verdict}")
        rows.append(dict(name=name, ok=ok, issues=issues, **detail))

    print("=" * 104)
    print(f"\n✅ 通过: {len(rows)-len(bad)}  |  ❌ 有问题: {len(bad)}  |  总计: {len(srts)}")
    if bad:
        print("\n问题清单（已排除休息/间歇段，以下是活跃语音占比高却没转出文字的真丢失）:")
        for n, r in bad: print(f"  ❌ {n}\n     {r}")
    with open(os.path.join(srt_dir, "_verify_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入: {os.path.join(srt_dir, '_verify_report.json')}")
    return 0 if not bad else 1

def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--single":
        return main_single()
    return main_dir()

sys.exit(main())
