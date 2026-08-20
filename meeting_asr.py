#!/usr/bin/env python3
"""
会议ASR流水线 — 区分发言人

用法:
  python /root/autodl-tmp/meeting_asr.py "会议录音.wav"               # 自动选择分割后端
  python /root/autodl-tmp/meeting_asr.py "会议录音.wav" --hf_token hf_xxxx   # pyannote（最准）
  python /root/autodl-tmp/meeting_asr.py "会议录音.wav" --num_speakers 3

流程:
  1. 说话人分割（三级降级: pyannote → speechbrain → silero+MFCC）
  2. VibeVoice-ASR 逐块转写（无音乐检测，最大1200s/块）
  3. 按时间段匹配：每条ASR字幕找对应说话人
  4. 输出 SRT（含[Speaker A]标签）+ JSON + TXT发言汇总

输出格式:
  1
  00:00:05,200 --> 00:00:12,400
  [Speaker A] 各位早上好，今天我们讨论一下季度规划。
"""

import os
import sys
import json
import time
import argparse
import numpy as np


# ====================================================================
# 工具函数
# ====================================================================

def fmt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def get_start(seg):
    return seg.get("start_time", seg.get("Start", 0))


def speaker_label(idx):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < 26:
        return f"Speaker {letters[idx]}"
    return f"Speaker {letters[idx // 26 - 1]}{letters[idx % 26]}"


# ====================================================================
# Step 1A: pyannote 说话人分割（需 HF token）
# ====================================================================

def run_diarization_pyannote(audio_path, hf_token, num_speakers=None,
                              min_speakers=None, max_speakers=None):
    from pyannote.audio import Pipeline
    import torch

    print("[Diarize/pyannote] 加载模型 pyannote/speaker-diarization-3.1 ...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers: kwargs["min_speakers"] = min_speakers
        if max_speakers: kwargs["max_speakers"] = max_speakers

    t0 = time.time()
    diarization = pipeline(audio_path, **kwargs)
    print(f"[Diarize/pyannote] 完成，耗时 {time.time()-t0:.1f}s")

    segments = [(turn.start, turn.end, spk)
                for turn, _, spk in diarization.itertracks(yield_label=True)]
    segments.sort(key=lambda x: x[0])
    return segments


# ====================================================================
# Step 1B: speechbrain ECAPA-TDNN 说话人分割（无需 HF token）
# ====================================================================

def run_diarization_speechbrain(audio_path, num_speakers=None, max_speakers=6):
    """
    用 speechbrain ECAPA-TDNN 提取说话人嵌入 + 层次聚类。
    speechbrain 模型从 HuggingFace 下载但不需要用户授权。
    """
    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    print("[Diarize/speechbrain] 加载 ECAPA-TDNN 说话人嵌入模型...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_dir = "/root/autodl-tmp/speechbrain_spk"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    print("[Diarize/speechbrain] 模型加载完成")

    # VAD: 用 silero-vad 检测语音段
    print("[Diarize/speechbrain] 运行 VAD (silero)...")
    vad_segments = _silero_vad(audio_path)
    if not vad_segments:
        print("[Diarize/speechbrain] VAD 未检测到语音，使用全段")
        waveform, sr = torchaudio.load(audio_path)
        vad_segments = [(0.0, waveform.shape[1] / sr)]

    # 过滤太短的段（<0.5s）
    vad_segments = [(s, e) for s, e in vad_segments if e - s >= 0.5]
    print(f"[Diarize/speechbrain] VAD 检测到 {len(vad_segments)} 个语音段")

    # 提取每段的说话人嵌入
    waveform, orig_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 如需重采样到16kHz
    target_sr = 16000
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    else:
        sr = orig_sr

    embeddings = []
    valid_segments = []
    print(f"[Diarize/speechbrain] 提取 {len(vad_segments)} 段的说话人嵌入...")
    for i, (s, e) in enumerate(vad_segments):
        s_sample = int(s * sr)
        e_sample = min(int(e * sr), waveform.shape[1])
        seg_wav = waveform[:, s_sample:e_sample]
        if seg_wav.shape[1] < int(0.3 * sr):
            continue
        # 取最多3秒的片段提取嵌入（避免OOM）
        max_samples = int(3.0 * sr)
        if seg_wav.shape[1] > max_samples:
            seg_wav = seg_wav[:, :max_samples]
        try:
            emb = classifier.encode_batch(seg_wav)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_segments.append((s, e))
        except Exception as ex:
            pass

    if len(embeddings) < 2:
        print("[Diarize/speechbrain] 嵌入数量不足，标记为单一说话人")
        return [(s, e, "SPEAKER_00") for s, e in valid_segments]

    emb_matrix = normalize(np.array(embeddings))

    # 确定聚类数
    if num_speakers is not None:
        n_clusters = num_speakers
    else:
        # 用 AHC + 距离阈值估算
        n_clusters = _estimate_n_speakers(emb_matrix, max_speakers)
    print(f"[Diarize/speechbrain] 聚类为 {n_clusters} 位说话人")

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(emb_matrix)

    # 构建分割结果
    segments = [(s, e, f"SPEAKER_{l:02d}")
                for (s, e), l in zip(valid_segments, labels)]
    segments.sort(key=lambda x: x[0])

    # 合并相邻同说话人段（间隔<1s的合并）
    merged = []
    for seg in segments:
        if merged and merged[-1][2] == seg[2] and seg[0] - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], seg[1], seg[2])
        else:
            merged.append(list(seg))
    segments = [tuple(s) for s in merged]

    return segments


def _silero_vad(audio_path, threshold=0.5, min_speech_ms=300, min_silence_ms=500):
    """用 silero-vad 检测语音段，返回 [(start_sec, end_sec), ...]"""
    import torch
    import torchaudio

    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, read_audio, *_) = utils

        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_speech_timestamps(
            wav, model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms
        )
        return [(t['start'] / 16000, t['end'] / 16000) for t in timestamps]
    except Exception as e:
        print(f"[VAD] silero-vad 失败: {e}，改用固定切分")
        waveform, sr = torchaudio.load(audio_path)
        total = waveform.shape[1] / sr
        # 每2秒一段作为兜底
        return [(i * 2.0, min((i + 1) * 2.0, total))
                for i in range(int(total / 2.0))]


def _estimate_n_speakers(emb_matrix, max_speakers=6):
    """用 AHC 距离变化估算说话人数"""
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    best_n = 2
    best_gap = -1
    n_max = min(max_speakers, len(emb_matrix) - 1)

    if n_max < 2:
        return 1

    prev_inertia = None
    inertias = []
    for n in range(2, n_max + 1):
        model = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
        labels = model.fit_predict(emb_matrix)
        # 计算类内平均距离
        intra = []
        for c in range(n):
            members = emb_matrix[labels == c]
            if len(members) > 1:
                center = members.mean(axis=0)
                dists = np.linalg.norm(members - center, axis=1)
                intra.append(dists.mean())
        inertia = np.mean(intra) if intra else 0
        inertias.append((n, inertia))

    # 找下降最大的拐点（肘部法则）
    if len(inertias) >= 2:
        gaps = [(inertias[i][1] - inertias[i+1][1])
                for i in range(len(inertias)-1)]
        best_idx = np.argmax(gaps)
        best_n = inertias[best_idx][0]

    return max(2, int(best_n))


# ====================================================================
# Step 1C: 主分割入口（自动降级）
# ====================================================================

def run_diarization(audio_path, hf_token=None, num_speakers=None,
                    min_speakers=None, max_speakers=None):
    """
    三级降级:
      1. pyannote（需 HF token）— 最准
      2. speechbrain ECAPA-TDNN（无需 token）— 中等
      3. silero VAD + MFCC聚类（纯 torch，最轻量）— 基础
    """
    # 策略1: pyannote
    if hf_token:
        try:
            print("[Diarize] 使用 pyannote（精度最高）")
            segs = run_diarization_pyannote(audio_path, hf_token,
                                            num_speakers, min_speakers, max_speakers)
            speakers = sorted(set(s[2] for s in segs))
            print(f"[Diarize] 检测到 {len(speakers)} 位说话人: {', '.join(speakers)}")
            return segs, speakers
        except Exception as e:
            print(f"[Diarize] pyannote 失败: {e}，降级到 speechbrain")

    # 策略2: speechbrain
    try:
        print("[Diarize] 使用 speechbrain ECAPA-TDNN（无需 HF token）")
        segs = run_diarization_speechbrain(audio_path, num_speakers, max_speakers or 6)
        speakers = sorted(set(s[2] for s in segs))
        print(f"[Diarize] 检测到 {len(speakers)} 位说话人: {', '.join(speakers)}")
        _print_speaker_stats(segs, speakers)
        return segs, speakers
    except Exception as e:
        print(f"[Diarize] speechbrain 失败: {e}，降级到基础 MFCC 聚类")

    # 策略3: silero VAD + MFCC聚类
    try:
        print("[Diarize] 使用 silero VAD + MFCC 聚类（基础模式）")
        segs = run_diarization_mfcc(audio_path, num_speakers or 2, max_speakers or 6)
        speakers = sorted(set(s[2] for s in segs))
        print(f"[Diarize] 检测到 {len(speakers)} 位说话人: {', '.join(speakers)}")
        _print_speaker_stats(segs, speakers)
        return segs, speakers
    except Exception as e:
        print(f"[Diarize] 所有分割方案失败: {e}")
        print("[Diarize] 将跳过说话人分割，所有字幕标记为 Unknown")
        return [], []


def _print_speaker_stats(segments, speakers):
    for spk in speakers:
        spk_segs = [s for s in segments if s[2] == spk]
        total = sum(e - s for s, e, _ in spk_segs)
        print(f"  {spk}: {len(spk_segs)} 段，共 {total/60:.1f} 分钟")


# ====================================================================
# Step 1D: 最轻量后备 — silero VAD + torchaudio MFCC 聚类
# ====================================================================

def run_diarization_mfcc(audio_path, num_speakers=2, max_speakers=6):
    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    print("[Diarize/MFCC] 运行 silero VAD...")
    vad_segs = _silero_vad(audio_path)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80}
    )

    embeddings = []
    valid_segs = []
    for s, e in vad_segs:
        if e - s < 0.5:
            continue
        s_s = int(s * sr)
        e_s = min(int(e * sr), waveform.shape[1])
        chunk = waveform[:, s_s:e_s]
        # 最多取3s
        if chunk.shape[1] > 3 * sr:
            chunk = chunk[:, :3 * sr]
        mfcc = mfcc_transform(chunk)  # (1, 40, T)
        emb = mfcc.mean(dim=-1).squeeze().numpy()  # (40,)
        embeddings.append(emb)
        valid_segs.append((s, e))

    if len(embeddings) < 2:
        return [(s, e, "SPEAKER_00") for s, e in valid_segs]

    emb_matrix = normalize(np.array(embeddings))
    n_clusters = num_speakers if num_speakers else _estimate_n_speakers(emb_matrix, max_speakers)

    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(emb_matrix)

    segments = [(s, e, f"SPEAKER_{l:02d}")
                for (s, e), l in zip(valid_segs, labels)]
    segments.sort(key=lambda x: x[0])

    # 合并相邻同说话人段
    merged = []
    for seg in segments:
        if merged and merged[-1][2] == seg[2] and seg[0] - merged[-1][1] < 1.5:
            merged[-1] = (merged[-1][0], seg[1], seg[2])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged]


# ====================================================================
# Step 2: ASR 转写（课程模式，无音乐检测）
# ====================================================================

def run_asr_course(audio_path, model_path, max_chunk_sec=1200, overlap_sec=5,
                   output_dir="/root/autodl-tmp/output"):
    import torch
    import torchaudio

    print(f"\n[ASR] 加载音频...")
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total_sec = waveform.shape[1] / sample_rate
    print(f"[ASR] 音频时长: {total_sec:.1f}s ({total_sec/60:.1f}分钟)")

    chunks = []
    offset = 0.0
    while offset < total_sec:
        dur = min(max_chunk_sec, total_sec - offset)
        if dur < 5:
            break
        chunks.append((offset, dur))
        offset += dur - overlap_sec
    print(f"[ASR] 切分为 {len(chunks)} 块")

    print(f"[ASR] 加载模型...")
    sys.path.insert(0, "/root/autodl-tmp/VibeVoice")
    from demo.vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference
    asr = VibeVoiceASRBatchInference(model_path)

    all_segments = []
    max_end_so_far = 0.0
    total_gen_time = 0

    for i, (offset, duration) in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"Chunk {i}/{len(chunks)-1}: "
              f"{int(offset//60):02d}:{int(offset%60):02d} → "
              f"{int((offset+duration)//60):02d}:{int((offset+duration)%60):02d}")

        start_sample = int(offset * sample_rate)
        end_sample = min(int((offset + duration) * sample_rate), waveform.shape[1])
        chunk_wav = waveform[:, start_sample:end_sample]

        tmp_path = f"/tmp/meeting_chunk_{i}.wav"
        torchaudio.save(tmp_path, chunk_wav, sample_rate)

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

            if results:
                segs = results[0].get("segments", [])
                added = 0
                for seg in segs:
                    s_t = seg.get("start_time", seg.get("Start", 0))
                    e_t = seg.get("end_time", seg.get("End", 0))
                    abs_s = s_t + offset
                    abs_e = e_t + offset
                    if abs_s < max_end_so_far - 1:
                        continue
                    new_seg = dict(seg)
                    if "start_time" in new_seg:
                        new_seg["start_time"] = round(abs_s, 3)
                        new_seg["end_time"] = round(abs_e, 3)
                    if "Start" in new_seg:
                        new_seg["Start"] = round(abs_s, 3)
                        new_seg["End"] = round(abs_e, 3)
                    all_segments.append(new_seg)
                    max_end_so_far = max(max_end_so_far, abs_e)
                    added += 1
                print(f"  {len(segs)} 段 ({added} 有效)，耗时 {gen_time:.1f}s")
            else:
                print(f"  无结果")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        torch.cuda.empty_cache()

    all_segments.sort(key=get_start)
    print(f"\n[ASR] 完成: {len(all_segments)} 条字幕，推理总耗时 {total_gen_time:.1f}s")
    return all_segments, total_sec


# ====================================================================
# Step 3: 合并说话人标签
# ====================================================================

def assign_speakers(asr_segments, diar_segments, speakers):
    spk_order = {}
    for _, _, spk in diar_segments:
        if spk not in spk_order:
            spk_order[spk] = speaker_label(len(spk_order))

    if spk_order:
        print(f"\n[Merge] 说话人标签映射:")
        for spk_id, label in spk_order.items():
            print(f"  {spk_id} → {label}")

    result = []
    unassigned = 0

    for seg in asr_segments:
        s_start = seg.get("start_time", seg.get("Start", 0))
        s_end = seg.get("end_time", seg.get("End", 0))
        s_dur = max(s_end - s_start, 0.001)

        spk_overlap = {}
        for d_start, d_end, spk in diar_segments:
            overlap = max(0, min(s_end, d_end) - max(s_start, d_start))
            if overlap > 0:
                spk_overlap[spk] = spk_overlap.get(spk, 0) + overlap

        new_seg = dict(seg)
        if spk_overlap:
            best_spk = max(spk_overlap, key=spk_overlap.get)
            new_seg["speaker_id"] = spk_order[best_spk]
            new_seg["speaker_raw"] = best_spk
            new_seg["speaker_overlap_ratio"] = round(spk_overlap[best_spk] / s_dur, 3)
        else:
            # 找最近的说话人
            min_gap = float("inf")
            nearest_spk = None
            for d_start, d_end, spk in diar_segments:
                gap = max(s_start - d_end, d_start - s_end, 0)
                if gap < min_gap:
                    min_gap = gap
                    nearest_spk = spk
            if nearest_spk and min_gap < 3.0:
                new_seg["speaker_id"] = spk_order[nearest_spk]
                new_seg["speaker_raw"] = nearest_spk
                new_seg["speaker_overlap_ratio"] = 0.0
            else:
                new_seg["speaker_id"] = "Unknown"
                new_seg["speaker_raw"] = "Unknown"
                new_seg["speaker_overlap_ratio"] = 0.0
                unassigned += 1

        result.append(new_seg)

    print(f"[Merge] 完成: {len(result)} 条字幕，{unassigned} 条无法确定说话人")
    return result, spk_order


# ====================================================================
# Step 4: 保存结果
# ====================================================================

def save_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get("start_time", seg.get("Start", 0))
            end = seg.get("end_time", seg.get("End", 0))
            text = seg.get("text", seg.get("Content", ""))
            speaker = seg.get("speaker_id", "")
            f.write(f"{i}\n")
            f.write(f"{fmt_time(start)} --> {fmt_time(end)}\n")
            prefix = f"[{speaker}] " if speaker and speaker != "Unknown" else ""
            f.write(f"{prefix}{text}\n\n")


def save_txt_summary(segments, spk_order, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("会议转写 — 发言人区分版\n")
        f.write("=" * 60 + "\n\n")

        prev_spk = None
        buffer = []
        seg_start = 0

        def flush(spk, buf, t):
            if not buf:
                return
            m, s = int(t // 60), int(t % 60)
            f.write(f"[{spk}] {m:02d}:{s:02d}\n")
            f.write("".join(buf).strip() + "\n\n")

        for seg in segments:
            spk = seg.get("speaker_id", "Unknown")
            text = seg.get("text", seg.get("Content", ""))
            t = seg.get("start_time", seg.get("Start", 0))
            if spk != prev_spk:
                flush(prev_spk, buffer, seg_start)
                buffer = [text + " "]
                prev_spk = spk
                seg_start = t
            else:
                buffer.append(text + " ")
        flush(prev_spk, buffer, seg_start)

        f.write("=" * 60 + "\n")
        f.write("发言统计\n")
        f.write("=" * 60 + "\n")
        spk_stats = {}
        for seg in segments:
            spk = seg.get("speaker_id", "Unknown")
            dur = (seg.get("end_time", seg.get("End", 0))
                   - seg.get("start_time", seg.get("Start", 0)))
            spk_stats[spk] = spk_stats.get(spk, 0) + max(dur, 0)

        total_speech = sum(spk_stats.values())
        for spk in sorted(spk_stats):
            dur = spk_stats[spk]
            pct = dur / total_speech * 100 if total_speech > 0 else 0
            f.write(f"  {spk}: {dur/60:.1f}分钟 ({pct:.0f}%)\n")


# ====================================================================
# 主流水线
# ====================================================================

def run_meeting_pipeline(audio_path, output_dir, model_path,
                         hf_token=None, num_speakers=None,
                         min_speakers=None, max_speakers=None,
                         max_chunk_sec=1200, overlap_sec=5):

    print(f"\n{'='*60}")
    print(f"会议ASR流水线 — 区分发言人")
    print(f"音频: {os.path.basename(audio_path)}")
    print(f"{'='*60}")

    # Step 1: 说话人分割
    print(f"\n>>> Step 1: 说话人分割")
    diar_segments, speakers = run_diarization(
        audio_path, hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    # Step 1.5: 清理 GPU 显存，防止 speechbrain 残留与 VibeVoice-ASR 冲突
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[GPU] 已清理显存，准备加载 ASR 模型")
    except Exception:
        pass

    # Step 2: ASR 转写
    print(f"\n>>> Step 2: ASR 转写")
    asr_segments, total_sec = run_asr_course(
        audio_path, model_path,
        max_chunk_sec=max_chunk_sec,
        overlap_sec=overlap_sec,
        output_dir=output_dir
    )

    # Step 3: 合并
    print(f"\n>>> Step 3: 合并说话人标签")
    merged_segments, spk_order = assign_speakers(asr_segments, diar_segments, speakers)

    # Step 4: 保存
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(audio_path))[0]

    srt_path = os.path.join(output_dir, f"{basename}.srt")
    json_path = os.path.join(output_dir, f"{basename}.json")
    txt_path = os.path.join(output_dir, f"{basename}_发言汇总.txt")

    save_srt(merged_segments, srt_path)
    save_txt_summary(merged_segments, spk_order, txt_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": audio_path,
            "mode": "meeting",
            "total_duration": total_sec,
            "speakers": list(spk_order.values()),
            "speaker_map": {v: k for k, v in spk_order.items()},
            "n_segments": len(merged_segments),
            "diarization_segments": len(diar_segments),
            "segments": merged_segments
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完成！{len(merged_segments)} 条字幕，{len(speakers)} 位说话人")
    print(f"  SRT:  {srt_path}")
    print(f"  TXT:  {txt_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*60}")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="会议ASR — 区分发言人")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output")
    parser.add_argument("--model_path", default="/root/autodl-tmp/VibeVoice-ASR")
    parser.add_argument("--hf_token", default=None, help="HuggingFace Token（pyannote用，可选）")
    parser.add_argument("--num_speakers", type=int, default=None, help="已知说话人数（提高精度）")
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--max_chunk", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=5)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    run_meeting_pipeline(
        audio_path=args.audio,
        output_dir=args.output_dir,
        model_path=args.model_path,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        max_chunk_sec=args.max_chunk,
        overlap_sec=args.overlap
    )
