#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/VibeVoice || exit 1
cp demo/vibevoice_asr_inference_from_file.py.bak_original demo/vibevoice_asr_inference_from_file.py
echo "[BENCH_START] $(date +%s)"
python demo/vibevoice_asr_inference_from_file.py   --model_path /root/autodl-tmp/VibeVoice-ASR   --audio_files /root/autodl-tmp/qc/bench/bench600.wav   --max_audio_duration 1200   --output_dir /root/autodl-tmp/qc/bench
echo "[BENCH_END] $(date +%s) rc=$?"
