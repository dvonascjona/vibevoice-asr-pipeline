#!/usr/bin/env python3
"""
智能直播ASR流水线 — 基于音乐检测的切分

用法:
  python /root/autodl-tmp/smart_live_asr.py "/root/autodl-tmp/2月7号20点2h12.MP3"

流程:
  1. 分析音频，检测音乐段位置
  2. 在音乐段中间切分（每个chunk = 一轮完整话术，不含音乐）
  3. 对每个chunk调用ASR模型转写
  4. 合并结果，输出SRT+JSON

直播结构:
  [话术~10min] → [音乐BGM~30s-1min] → [话术] → [音乐] → ...
  切分点选在音乐段中间，这样每个chunk都是纯说话内容
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ====================================================================
# 1. 音乐检测
# ====================================================================

def detect_music_segments(audio_path, analysis_sr=8000, window_sec=3.0,
                          frame_sec=0.05, min_music_sec=30):
    """
    检测音频中的音乐段。

    原理: 在3秒窗口内计算50ms能量帧的变异系数(CV)
      说话: CV高（有微停顿，能量波动大）
      音乐: CV低（连续声音，能量稳定）
    """
    import librosa

    print("[SmartSplit] 正在分析音频，检测音乐段...")
    y, sr = librosa.load(audio_path, sr=analysis_sr, mono=True)
    total_sec = len(y) / sr
    print(f"[SmartSplit] 音频时长: {total_sec:.1f}s ({total_sec/60:.1f}分钟)")

    # 50ms帧的RMS能量
    hop = int(sr * frame_sec)
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=hop * 2)[0]

    # 每个窗口的能量变异系数
    frames_per_window = int(window_sec / frame_sec)
    n_windows = len(rms) // frames_per_window

    cv_values = []
    energy_values = []
    for i in range(n_windows):
        start = i * frames_per_window
        end = start + frames_per_window
        seg = rms[start:end]
        mean_e = float(np.mean(seg))
        energy_values.append(mean_e)
        cv = float(np.std(seg) / mean_e) if mean_e > 1e-6 else 0.0
        cv_values.append(cv)

    cv_arr = np.array(cv_values)
    energy_arr = np.array(energy_values)

    # 自适应阈值
    active = energy_arr > np.percentile(energy_arr[energy_arr > 0], 10) if np.any(energy_arr > 0) else energy_arr > 0
    if np.any(cv_arr[active] > 0):
        cv_threshold = np.percentile(cv_arr[active], 30)
    else:
        cv_threshold = 0.3

    # 分类: 低CV且有能量 = 音乐, 低能量 = 静音(也作为分割点)
    energy_threshold = np.percentile(energy_arr[energy_arr > 0], 12) if np.any(energy_arr > 0) else 0
    is_non_speech = (cv_arr < cv_threshold) | (energy_arr < energy_threshold)

    # 中值滤波平滑
    smoothed = is_non_speech.copy()
    k = 2
    for i in range(k, len(smoothed) - k):
        window = is_non_speech[i - k:i + k + 1]
        smoothed[i] = np.sum(window) > len(window) // 2

    # 提取连续的非说话段
    segments = []
    in_seg = False
    seg_start = 0

    for i in range(len(smoothed)):
        if smoothed[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not smoothed[i] and in_seg:
            dur = (i - seg_start) * window_sec
            if dur >= min_music_sec:
                segments.append((seg_start * window_sec, i * window_sec))
            in_seg = False

    if in_seg:
        dur = (len(smoothed) - seg_start) * window_sec
        if dur >= min_music_sec:
            segments.append((seg_start * window_sec, len(smoothed) * window_sec))

    print(f"[SmartSplit] 检测到 {len(segments)} 个音乐/非说话段:")
    for i, (s, e) in enumerate(segments):
        print(f"  Music {i + 1}: {int(s // 60):02d}:{int(s % 60):02d} → {int(e // 60):02d}:{int(e % 60):02d} ({e - s:.0f}秒)")

    return segments, total_sec


# ====================================================================
# 2. 智能切分
# ====================================================================

def compute_smart_chunks(music_segments, total_sec, max_chunk_sec=900, overlap_sec=5):
    """
    基于音乐段位置计算切分点。

    切分策略:
      - 切分点 = 每个音乐段的中间位置
      - 如果相邻音乐段间距 > max_chunk_sec，在中间额外加切分点
      - 每个chunk = 一轮完整话术（从上一个音乐段中间到下一个音乐段中间）
    """
    if not music_segments:
        # 没检测到音乐，按固定间隔切
        print("[SmartSplit] 未检测到音乐段，使用固定切分")
        chunks = []
        offset = 0.0
        while offset < total_sec:
            dur = min(max_chunk_sec, total_sec - offset)
            if dur < 10:
                break
            chunks.append((offset, dur))
            offset += dur - overlap_sec
        return chunks

    # 切分点 = 音乐段中间
    split_points = [(s + e) / 2 for s, e in music_segments]
    boundaries = [0.0] + split_points + [total_sec]

    chunks = []
    for i in range(len(boundaries) - 1):
        cs = boundaries[i]
        ce = boundaries[i + 1]
        dur = ce - cs

        if dur < 8:  # 太短跳过
            continue

        if dur > max_chunk_sec:
            # 太长，细分
            n = int(np.ceil(dur / max_chunk_sec))
            part = dur / n
            for j in range(n):
                ps = cs + j * part
                pd = min(part + overlap_sec, total_sec - ps)
                chunks.append((max(0, ps - (overlap_sec if j > 0 else 0)), pd + (overlap_sec if j > 0 else 0)))
        else:
            # 加overlap
            actual_start = max(0, cs - overlap_sec) if i > 0 else cs
            actual_dur = min(ce - actual_start + overlap_sec, total_sec - actual_start)
            chunks.append((actual_start, actual_dur))

    print(f"\n[SmartSplit] 智能切分结果: {len(chunks)} 个chunks")
    for i, (offset, dur) in enumerate(chunks):
        s_m, s_s = int(offset // 60), int(offset % 60)
        e_m, e_s = int((offset + dur) // 60), int((offset + dur) % 60)
        print(f"  Chunk {i}: {s_m:02d}:{s_s:02d} → {e_m:02d}:{e_s:02d} ({dur:.0f}秒)")

    return chunks


# ====================================================================
# 3. ASR处理
# ====================================================================

def run_asr_pipeline(audio_path, output_dir="/root/autodl-tmp/output",
                     model_path="/root/autodl-tmp/VibeVoice-ASR",
                     max_chunk_sec=900, overlap_sec=5):
    """完整的智能ASR流水线"""
    import torch
    import torchaudio

    # Step 1: 检测音乐
    music_segments, total_sec = detect_music_segments(audio_path)

    # Step 2: 计算切分点
    chunks = compute_smart_chunks(music_segments, total_sec, max_chunk_sec, overlap_sec)

    # Step 3: 加载音频
    print(f"\n[ASR] 加载音频...")
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 4: 加载ASR模型
    print(f"[ASR] 加载模型...")
    sys.path.insert(0, "/root/autodl-tmp/VibeVoice")
    from demo.vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference

    asr = VibeVoiceASRBatchInference(model_path)

    # Step 5: 逐chunk处理
    all_segments = []
    total_gen_time = 0
    max_end_so_far = 0.0  # 已收录字幕的最大结束时间（绝对值）

    for i, (offset, duration) in enumerate(chunks):
        print(f"\n{'=' * 60}")
        print(f"Chunk {i}/{len(chunks) - 1}: offset={offset:.1f}s ({int(offset // 60):02d}:{int(offset % 60):02d}), duration={duration:.0f}s")
        print(f"{'=' * 60}")

        # 提取chunk音频
        start_sample = int(offset * sample_rate)
        end_sample = min(int((offset + duration) * sample_rate), waveform.shape[1])
        chunk_wav = waveform[:, start_sample:end_sample]

        # 保存临时文件
        tmp_path = f"/tmp/smart_chunk_{i}.wav"
        torchaudio.save(tmp_path, chunk_wav, sample_rate)

        # 运行ASR
        try:
            t0 = time.time()
            results = asr.transcribe_with_batching(
                audio_inputs=[tmp_path],
                batch_size=1,
                max_new_tokens=32768,
                do_sample=False
            )
            gen_time = time.time() - t0
            total_gen_time += gen_time

            if results and len(results) > 0:
                result = results[0]
                segments = result.get("segments", [])

                added = 0
                for seg in segments:
                    seg_start = seg.get("start_time", seg.get("Start", 0))
                    seg_end = seg.get("end_time", seg.get("End", 0))
                    abs_start = seg_start + offset
                    abs_end = seg_end + offset

                    # 跳过已被前一个chunk覆盖的重复内容
                    if abs_start < max_end_so_far - 1:
                        continue

                    # 应用偏移
                    new_seg = dict(seg)
                    if "start_time" in new_seg:
                        new_seg["start_time"] = round(abs_start, 3)
                        new_seg["end_time"] = round(abs_end, 3)
                    if "Start" in new_seg:
                        new_seg["Start"] = round(abs_start, 3)
                        new_seg["End"] = round(abs_end, 3)

                    all_segments.append(new_seg)
                    max_end_so_far = max(max_end_so_far, abs_end)
                    added += 1

                print(f"  Chunk {i}: {len(segments)} segments ({added} added, {len(segments)-added} deduped), {gen_time:.1f}s")
            else:
                print(f"  Chunk {i}: No results")

        except Exception as e:
            print(f"  Chunk {i} ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 清理GPU显存
        torch.cuda.empty_cache()

    # Step 6: 排序
    def get_start(seg):
        return seg.get("start_time", seg.get("Start", 0))
    all_segments.sort(key=get_start)

    # Step 7: 保存结果
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(audio_path))[0]

    # SRT
    srt_path = os.path.join(output_dir, f"{basename}.srt")
    save_srt(all_segments, srt_path)

    # JSON
    json_path = os.path.join(output_dir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": audio_path,
            "total_duration": total_sec,
            "generation_time": total_gen_time,
            "music_segments": [{"start": s, "end": e} for s, e in music_segments],
            "n_chunks": len(chunks),
            "segments": all_segments
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"完成! {len(all_segments)} 条字幕")
    print(f"  SRT: {srt_path}")
    print(f"  JSON: {json_path}")
    print(f"  总推理时间: {total_gen_time:.1f}s")
    print(f"{'=' * 60}")


def save_srt(segments, path):
    """保存SRT字幕文件"""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get("start_time", seg.get("Start", 0))
            end = seg.get("end_time", seg.get("End", 0))
            text = seg.get("text", seg.get("Content", ""))
            speaker = seg.get("speaker_id", seg.get("Speaker", None))

            sh, sm, ss = int(start // 3600), int((start % 3600) // 60), start % 60
            eh, em, es = int(end // 3600), int((end % 3600) // 60), end % 60

            f.write(f"{i}\n")
            f.write(f"{sh:02d}:{sm:02d}:{ss:06.3f} --> {eh:02d}:{em:02d}:{es:06.3f}\n".replace(".", ","))
            prefix = f"[{speaker}] " if speaker is not None else ""
            f.write(f"{prefix}{text}\n\n")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能直播ASR - 基于音乐检测的切分")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output", help="输出目录")
    parser.add_argument("--model_path", default="/root/autodl-tmp/VibeVoice-ASR", help="模型路径")
    parser.add_argument("--max_chunk", type=int, default=900, help="单chunk最大秒数(默认900=15min)")
    parser.add_argument("--overlap", type=int, default=5, help="chunk重叠秒数")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    run_asr_pipeline(
        audio_path=args.audio,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_chunk_sec=args.max_chunk,
        overlap_sec=args.overlap
    )
