#!/usr/bin/env python3
"""
说话人分割 — 独立运行，结果存 JSON
用法: python diarize_only.py "音频文件.wav" [--num_speakers 2]
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

def speaker_label(idx):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"Speaker {letters[idx]}" if idx < 26 else f"Speaker {letters[idx//26-1]}{letters[idx%26]}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output")
    parser.add_argument("--num_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=6)
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Diarize] 音频: {os.path.basename(args.audio)}")

    # ── VAD ─────────────────────────────────────────────────────
    import torch, torchaudio
    print("[Diarize] 运行 silero-vad...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', model='silero_vad',
        force_reload=False, trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_) = utils
    wav = read_audio(args.audio, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, vad_model, threshold=0.5,
        sampling_rate=16000, min_speech_duration_ms=300, min_silence_duration_ms=500)
    vad_segs = [(t['start']/16000, t['end']/16000) for t in timestamps]
    vad_segs = [(s, e) for s, e in vad_segs if e - s >= 0.5]
    print(f"[Diarize] VAD: {len(vad_segs)} 个语音段")

    # 释放 VAD
    del vad_model, wav
    torch.cuda.empty_cache()

    # ── 说话人嵌入 ───────────────────────────────────────────────
    print("[Diarize] 加载 speechbrain ECAPA-TDNN (CPU)...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/root/autodl-tmp/speechbrain_spk",
        run_opts={"device": "cpu"}   # 强制CPU，不碰GPU
    )

    waveform, orig_sr = torchaudio.load(args.audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_sr, 16000)(waveform)
    sr = 16000

    import numpy as np
    embeddings, valid_segs = [], []
    print(f"[Diarize] 提取 {len(vad_segs)} 段嵌入...")
    for s, e in vad_segs:
        s_s, e_s = int(s * sr), min(int(e * sr), waveform.shape[1])
        chunk = waveform[:, s_s:e_s]
        if chunk.shape[1] < int(0.3 * sr):
            continue
        if chunk.shape[1] > 3 * sr:
            chunk = chunk[:, :3 * sr]
        try:
            emb = clf.encode_batch(chunk)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_segs.append((s, e))
        except:
            pass

    # 释放 speechbrain
    del clf, waveform
    import gc; gc.collect()

    if len(embeddings) < 2:
        print("[Diarize] 嵌入不足，标记为单一说话人")
        segs = [(s, e, "SPEAKER_00") for s, e in valid_segs]
    else:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize
        emb_matrix = normalize(np.array(embeddings))

        if args.num_speakers:
            n = args.num_speakers
        else:
            # 肘部法估算
            best_n, best_gap = 2, -1
            for n in range(2, min(args.max_speakers, len(emb_matrix))):
                labels = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average").fit_predict(emb_matrix)
                intra = []
                for c in range(n):
                    m = emb_matrix[labels == c]
                    if len(m) > 1:
                        intra.append(np.linalg.norm(m - m.mean(0), axis=1).mean())
                score = np.mean(intra) if intra else 0
                if n > 2 and (prev_score - score) > best_gap:
                    best_gap = prev_score - score
                    best_n = n - 1
                prev_score = score
            n = best_n

        print(f"[Diarize] 聚类为 {n} 位说话人")
        labels = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average").fit_predict(emb_matrix)

        raw_segs = [(s, e, f"SPEAKER_{l:02d}") for (s, e), l in zip(valid_segs, labels)]
        raw_segs.sort(key=lambda x: x[0])

        # 合并相邻同说话人段
        segs = []
        for seg in raw_segs:
            if segs and segs[-1][2] == seg[2] and seg[0] - segs[-1][1] < 1.0:
                segs[-1] = (segs[-1][0], seg[1], seg[2])
            else:
                segs.append(list(seg))
        segs = [tuple(s) for s in segs]

    # 统计
    speakers = sorted(set(s[2] for s in segs))
    spk_map = {}
    for sp in segs:
        if sp[2] not in spk_map:
            spk_map[sp[2]] = speaker_label(len(spk_map))

    print(f"\n[Diarize] 结果: {len(speakers)} 位说话人")
    for spk in speakers:
        sp_segs = [s for s in segs if s[2] == spk]
        total = sum(e - s for s, e, _ in sp_segs)
        print(f"  {spk} → {spk_map[spk]}: {len(sp_segs)} 段, {total/60:.1f} 分钟")

    # 保存
    basename = os.path.splitext(os.path.basename(args.audio))[0]
    out_path = os.path.join(args.output_dir, f"{basename}_diar.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "audio": args.audio,
            "speakers": list(spk_map.values()),
            "speaker_map": spk_map,
            "segments": [{"start": s, "end": e, "speaker_raw": sp, "speaker": spk_map[sp]}
                         for s, e, sp in segs]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Diarize] 分割结果已保存: {out_path}")
    print("下一步: 跑 ASR 转写，再用 merge_speaker.py 合并标签")

if __name__ == "__main__":
    main()
