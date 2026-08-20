#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 真实验收：用 ffprobe 取时长（mutagen 对本批 wav 返回 length=None，不可用），
对缺口段用 ffmpeg volumedetect 实测是否静音，避免把"开课前空等待"误判为丢内容。

判定：
  尾部缺口 > TAIL_GAP  -> 实测该段音量；有语音=FAIL(模型偷懒截断)，静音=OK
  开头缺口 > HEAD_GAP  -> 同上
  中间空洞 > MID_GAP   -> 同上

用法:
  目录模式: python3 verify.py <srt_dir> <audio_dir1> [audio_dir2 ...]
  单文件模式: python3 verify.py --single <srt_path> <audio_path>   (exit 0=OK, 1=FAIL)
"""
import os, re, sys, glob, json, subprocess

HEAD_GAP = 60.0
TAIL_GAP = 120.0
MID_GAP  = 120.0
SILENCE_MAX_DB = -35.0   # 实测: 静音段 max≈-45dB, 语音段 max≈-17~-26dB

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

def dur(path):
    r = sh(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path])
    try: return float(r.stdout.strip())
    except: return 0.0

def max_db(path, ss, t):
    """返回该段 max_volume(dB)。取不到返回 None。"""
    if t <= 0.5: return None
    r = sh(["ffmpeg","-hide_banner","-nostats","-ss",f"{ss:.2f}","-t",f"{t:.2f}",
            "-i",path,"-af","volumedetect","-f","null","-"])
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr + r.stdout)
    return float(m.group(1)) if m else None

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

    issues, checks = [], []
    if audio_path:
        if head > HEAD_GAP:
            checks.append(("开头", head, max_db(audio_path, 0, min(head, 900))))
        if tail > TAIL_GAP:
            checks.append(("尾部", tail, max_db(audio_path, last, min(tail, 900))))
        if mid > MID_GAP:
            checks.append(("中间", mid, max_db(audio_path, mid_at, min(mid, 900))))

    for where, gap, db in checks:
        if db is None:
            issues.append(f"{where}缺{fmt(gap)}(未能实测)")
        elif db > SILENCE_MAX_DB:
            issues.append(f"{where}缺{fmt(gap)}有语音({db:.0f}dB)")

    detail = dict(audio_sec=round(adur,1), srt_end=round(last,1), subs=len(segs),
                  head=round(head,1), tail=round(tail,1), mid=round(mid,1))
    return (len(issues) == 0), issues, detail

def main_single():
    srt_path, audio_path = sys.argv[2], sys.argv[3]
    ok, issues, detail = check_one(srt_path, audio_path if os.path.isfile(audio_path) else None)
    if ok:
        print(f"OK 覆盖完整 (字幕{detail['subs']}条, 音频{fmt(detail['audio_sec'])}, 止于{fmt(detail['srt_end'])})")
        return 0
    else:
        print(f"FAIL {' + '.join(issues)}")
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
        print("\n问题清单（已排除静音段，以下是实测有语音却没转出文字的）:")
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
