#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export PYTHONIOENCODING=utf-8
# ============================================================
# ASR通用工具 2/2：转写 + 自动验收（三模式）
#
# 用法：
#   bash asr_run_and_verify.sh [course|live|meeting] [音频目录] [输出目录]
#
# 模式：
#   course（默认）— 逐文件直接跑ASR，适合课程/讲座（无BGM）
#   live          — 智能切分（音乐边界检测），适合直播回放（有BGM）
#   meeting       — 说话人分割+ASR，适合多人会议录音（区分发言人）
#
# 目录规范：
#   /root/autodl-tmp/asr/
#   ├── input/    ← 放待转写的音频（wav/mp3）
#   ├── output/   ← ASR产出（srt/json/txt）
#   └── done/     ← 转写完成的音频自动移入
#
# 流程：input放音频 → 跑脚本 → srt进output → 音频移到done
# 下次新任务：清空input丢新文件，output和done不会混
#
# 会议模式额外参数（通过环境变量设置）：
#   export HF_TOKEN=hf_xxxx          # pyannote 需要 HuggingFace Token
#   export MEETING_NUM_SPEAKERS=3    # 已知说话人数（留空自动检测）
#   export MEETING_MAX_SPEAKERS=5    # 最多说话人数上限
#
# 铁律：
#   - 模型会偷懒（注意力衰减），验收是标配不是事后补丁
#   - 质量检查用 ffprobe 实测时长 + volumedetect 实测缺口段是否有语音，
#     不靠 mutagen（对本批 wav 返回 None）也不靠尾部告别词
#   - 文件名有空格/特殊符号必须双引号传参
#   - 直播音频必须用live模式，否则音乐段会导致幻觉
#   - 会议录音用meeting模式，课程模式无法区分发言人
# ============================================================

set +e

# ===== 参数解析 =====
MODE="course"
ARG1="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

# 判断第一个参数是模式还是目录
if [ "$ARG1" = "course" ] || [ "$ARG1" = "live" ] || [ "$ARG1" = "meeting" ]; then
    MODE="$ARG1"
    AUDIO_DIR_ARG="$ARG2"
    OUTPUT_DIR_ARG="$ARG3"
else
    AUDIO_DIR_ARG="$ARG1"
    OUTPUT_DIR_ARG="$ARG2"
fi

# ===== 目录配置 =====
ASR_ROOT="/root/autodl-tmp/qc"
AUDIO_DIR="${AUDIO_DIR_ARG:-$ASR_ROOT/input}"
OUTPUT_DIR="${OUTPUT_DIR_ARG:-$ASR_ROOT/output}"
DONE_DIR="$ASR_ROOT/done"
VERIFY_SCRIPT="$ASR_ROOT/verify.py"

# ASR模型配置
MODEL_PATH="/root/autodl-tmp/VibeVoice-ASR"
ASR_SCRIPT="demo/vibevoice_asr_inference_from_file.py"
ASR_DIR="/root/autodl-tmp/VibeVoice"
SMART_LIVE_SCRIPT="/root/autodl-tmp/smart_live_asr.py"
MAX_DURATION=1200

if [ ! -d "$AUDIO_DIR" ]; then
    echo "错误: 音频目录不存在 $AUDIO_DIR"
    exit 1
fi

if [ ! -f "$VERIFY_SCRIPT" ]; then
    echo "错误: 验收脚本不存在 $VERIFY_SCRIPT"
    exit 1
fi

# ===== Step 1: 环境准备 =====
echo "=========================================="
echo "Step 1: 环境准备"
echo "=========================================="
source /etc/network_turbo 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" "$DONE_DIR"
echo "模式:     $MODE"
echo "输入目录: $AUDIO_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "完成目录: $DONE_DIR"

# 模式切换：课程模式用原始脚本，直播模式用反幻觉补丁版
ASR_SCRIPT_ORIGINAL="$ASR_DIR/demo/vibevoice_asr_inference_from_file.py.bak_original"
ASR_SCRIPT_PATCHED="$ASR_DIR/demo/vibevoice_asr_inference_from_file.py.bak"
ASR_SCRIPT_CURRENT="$ASR_DIR/demo/vibevoice_asr_inference_from_file.py"
MEETING_SCRIPT="/root/autodl-tmp/meeting_asr.py"

if [ "$MODE" = "course" ] && [ -f "$ASR_SCRIPT_ORIGINAL" ]; then
    cp "$ASR_SCRIPT_ORIGINAL" "$ASR_SCRIPT_CURRENT"
    echo "课程模式：已恢复原始ASR脚本（禁用反幻觉检测）"
elif [ "$MODE" = "live" ] && [ -f "$ASR_SCRIPT_PATCHED" ]; then
    cp "$ASR_SCRIPT_PATCHED" "$ASR_SCRIPT_CURRENT"
    echo "直播模式：已启用反幻觉补丁"
elif [ "$MODE" = "meeting" ] && [ -f "$ASR_SCRIPT_PATCHED" ]; then
    cp "$ASR_SCRIPT_PATCHED" "$ASR_SCRIPT_CURRENT"
    echo "会议模式：已启用反幻觉补丁"
fi

# live模式检查smart_live_asr.py是否存在
if [ "$MODE" = "live" ] && [ ! -f "$SMART_LIVE_SCRIPT" ]; then
    echo "错误: live模式需要 $SMART_LIVE_SCRIPT"
    echo "请将 smart_live_asr.py 上传到 /root/autodl-tmp/"
    exit 1
fi

# meeting模式检查meeting_asr.py是否存在
if [ "$MODE" = "meeting" ] && [ ! -f "$MEETING_SCRIPT" ]; then
    echo "错误: meeting模式需要 $MEETING_SCRIPT"
    echo "请将 meeting_asr.py 上传到 /root/autodl-tmp/"
    exit 1
fi

# ===== Step 2: ASR转写 =====
echo ""
echo "=========================================="
if [ "$MODE" = "live" ]; then
    echo "Step 2: ASR转写 — 直播模式（智能切分）"
elif [ "$MODE" = "meeting" ]; then
    echo "Step 2: ASR转写 — 会议模式（说话人分割+区分发言人）"
else
    echo "Step 2: ASR转写 — 课程模式（逐文件）"
fi
echo "=========================================="

cd "$ASR_DIR" || { echo "错误: ASR目录不存在 $ASR_DIR"; exit 1; }

count=0
skipped=0
success=0
failed=0
total=$(find -L "$AUDIO_DIR" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.MP3" -o -name "*.flac" -o -name "*.m4a" \) | wc -l)

echo "待处理: $total 个文件"
echo ""

for f in "$AUDIO_DIR"/*.wav "$AUDIO_DIR"/*.mp3 "$AUDIO_DIR"/*.MP3 "$AUDIO_DIR"/*.flac "$AUDIO_DIR"/*.m4a; do
    [ -f "$f" ] || continue
    basename_full=$(basename "$f")
    basename_noext="${basename_full%.*}"
    srt_file="$OUTPUT_DIR/${basename_noext}.srt"
    json_file="$OUTPUT_DIR/${basename_noext}.json"

    count=$((count + 1))

    # 断点续跑
    if [ -f "$srt_file" ]; then
        echo "[$count/$total] ⏭️  跳过: $basename_full"
        skipped=$((skipped + 1))
        continue
    fi

    echo ""
    echo "[$count/$total] 🎙️ 转写: $basename_full"
    echo "---"

    if [ "$MODE" = "live" ]; then
        # 直播模式：智能切分（音乐检测→切分→ASR→去重）
        python "$SMART_LIVE_SCRIPT" "$f" \
            --output_dir "$OUTPUT_DIR" \
            --model_path "$MODEL_PATH"
    elif [ "$MODE" = "meeting" ]; then
        # 会议模式：说话人分割 + ASR + 合并标签
        MEETING_ARGS="$f --output_dir $OUTPUT_DIR --model_path $MODEL_PATH"
        if [ -n "${HF_TOKEN:-}" ]; then
            MEETING_ARGS="$MEETING_ARGS --hf_token $HF_TOKEN"
        fi
        if [ -n "${MEETING_NUM_SPEAKERS:-}" ]; then
            MEETING_ARGS="$MEETING_ARGS --num_speakers $MEETING_NUM_SPEAKERS"
        fi
        if [ -n "${MEETING_MAX_SPEAKERS:-}" ]; then
            MEETING_ARGS="$MEETING_ARGS --max_speakers $MEETING_MAX_SPEAKERS"
        fi
        python "$MEETING_SCRIPT" $MEETING_ARGS
    else
        # 课程模式：直接跑ASR
        python "$ASR_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --audio_files "$f" \
            --max_audio_duration "$MAX_DURATION" \
            --output_dir "$OUTPUT_DIR"
    fi

    if [ -f "$srt_file" ]; then
        # 转完立刻验收：ffprobe实测时长 + volumedetect实测缺口段是否有语音
        verify_out=$(python3 "$VERIFY_SCRIPT" --single "$srt_file" "$f" 2>&1)
        verify_rc=$?

        if [ $verify_rc -eq 0 ]; then
            echo "✅ $verify_out → 移入done/"
            mv "$f" "$DONE_DIR/"
            success=$((success + 1))
        else
            echo "❌ $verify_out ，留在input待重跑"
            rm -f "$srt_file" "$json_file"
            failed=$((failed + 1))
        fi
    else
        echo "❌ 转写失败（留在input待重跑）"
        failed=$((failed + 1))
    fi
done

echo ""
echo "转写完成: 成功=$success 跳过=$skipped 失败=$failed / 总计=$total"

# ===== Step 3: 验收 =====
echo ""
echo "=========================================="
echo "Step 3: 验收 — 音频时长 vs SRT时间戳（含中间空洞检测）"
echo "=========================================="

python3 "$VERIFY_SCRIPT" "$OUTPUT_DIR" "$DONE_DIR" "$AUDIO_DIR"
