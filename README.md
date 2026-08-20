# VibeVoice-ASR Pipeline

基于微软 [VibeVoice-ASR](https://github.com/microsoft/VibeVoice) 模型的音频批量转写管线（三模式 + 真实验收），部署环境为 AutoDL GPU 服务器（RTX 5090）。

本仓库**不包含模型权重**（~17GB，MIT license，来自 HuggingFace），只包含转写/验收脚本。权重下载方式见下方。

## 模型权重

```bash
pip install -U huggingface_hub
huggingface-cli download microsoft/VibeVoice-ASR --local-dir ./VibeVoice-ASR
```

若上述 repo id 变更，以模型主页 [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) 上的最新链接为准。

推理代码本体（`demo/vibevoice_asr_inference_from_file.py`）需从 [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) 仓库获取，本仓库脚本只负责调度它。

## 三模式

| 模式 | 适用场景 | 切分策略 | 额外产出 |
|------|----------|----------|----------|
| `course` | 课程/讲座（无BGM） | 逐文件，max 1200s | SRT + JSON |
| `live` | 直播回放（有BGM） | 音乐边界智能切分（`smart_live_asr.py`） | SRT + JSON |
| `meeting` | 多人会议录音 | pyannote说话人分割+ASR（`meeting_asr.py`） | SRT[Speaker A/B] + TXT发言汇总 + JSON |

## 目录结构

```
qc/
├── run_asr.sh   转写主入口，三模式统一调度 + 逐文件自动验收
├── verify.py    真实验收脚本（ffprobe实测时长 + volumedetect实测缺口段是否有语音）
├── extract.sh   mp4 → 16k单声道wav 抽轨（带时长校验，差值>2s判失败）
└── bench.sh     速度基准测试

smart_live_asr.py            直播智能切分引擎（音乐边界检测CV）
meeting_asr.py                会议模式（说话人分割+ASR+标签合并）
patch_anti_hallucination.py   三层防幻觉补丁（repetition_penalty+RepetitionStopCriteria+trim_repetitive_tail）
diarize_only.py               仅说话人分割（不转写）
merge_speaker.py              说话人标签合并工具
```

## 用法

```bash
export PATH=/root/miniconda3/bin:$PATH   # AutoDL环境需要，screen非登录shell不加载conda
bash qc/run_asr.sh course /path/to/input_dir /path/to/output_dir
```

- `course`/`live`/`meeting` 三选一，默认 `course`
- 断点续跑：输出目录已有同名 `.srt` 自动跳过
- 转写成功的音频自动移入 `done/`；验收不通过的留在 `input/` 等重跑
- `live` 模式需要设置反幻觉补丁；`meeting` 模式需要 `HF_TOKEN`（pyannote/speaker-diarization-3.1 需接受条款）

meeting 模式额外环境变量：

```bash
export HF_TOKEN=hf_xxxx
export MEETING_NUM_SPEAKERS=3      # 已知说话人数时精度更高，留空自动检测
export MEETING_MAX_SPEAKERS=5
```

## 验收原则

**核心问题：`mutagen` 库对本项目常见的 wav 格式返回 `duration=None`**，导致覆盖率计算分母为0，历史上出现过"覆盖率恒为0%全部误判失败"的问题。`verify.py` 改用 `ffprobe` 取时长，并对每个疑似缺口段用 `ffmpeg volumedetect` 实测音量——只有"缺口段实测有语音（max_volume > -35dB）"才判失败，避免把"开场前静音等待"误判为模型丢内容。

同时检查**开头/尾部/中间空洞**三处（早期版本只查首尾，漏掉过中间大段真实内容缺失）。

```bash
# 单文件验收
python3 qc/verify.py --single output.srt input.wav   # exit 0=OK, 1=FAIL

# 目录全量验收
python3 qc/verify.py <srt目录> <音频目录1> [音频目录2 ...]
```

## License

脚本本身 MIT。VibeVoice-ASR 模型权重及推理代码遵循 [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) 仓库的 license（MIT）。
