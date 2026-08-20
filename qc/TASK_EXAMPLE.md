# 千川课程 ASR 转写任务

建立时间：2026-08-20（北京时间）
机器：AutoDL `connect.westc.seetacloud.com:30683`，RTX 5090 D 32G

## 任务范围

源目录 `/root/autodl-fs/qianchuan`，42 个文件，总时长 **67h51m**（244262 秒）：

| 类型 | 数量 | 说明 |
|---|---|---|
| wav | 25 | 已是 16kHz 单声道，直接可用（软链接进 input，源文件不动） |
| mp4 | 17 | 44.1kHz 立体声 aac，已用 ffmpeg 抽成 16k 单声道 wav |

模式：**course**（课程录播，逐文件直跑，用户确认）

## 目录

```
/root/autodl-tmp/qc/
├── input/      42 个（25 软链接 → autodl-fs 源文件 + 17 抽取的 wav）
├── done/       转写并验收通过后移入（软链接被移动，源文件不受影响）
├── logs/       asr_main.log 主日志 / extract.log 抽轨日志 / bench.log 测速
├── run_asr.sh  执行脚本（改自 /root/autodl-tmp/asr/asr_run_and_verify.sh）
├── extract.sh  mp4 抽轨（带时长校验，差值 >2s 判失败）
└── verify.py   真实验收脚本

/root/autodl-fs/qianchuan_asr/    ← 产出（文件存储，实例关机不丢）
├── *.srt  *.json
└── _verify_report.json           验收明细
```

## run_asr.sh 相对原脚本的 3 处改动

| 行 | 改动 | 原因 |
|---|---|---|
| 2 | 新增 `export PATH=/root/miniconda3/bin:$PATH` | screen 里非登录 shell 不加载 conda，原脚本报 `python: command not found` |
| 55 | `ASR_ROOT` → `/root/autodl-tmp/qc` | 原值 `/root/autodl-tmp/asr` 会把本批文件混进历史 done/（已有 821 个文件） |
| 134 | `find` → `find -L` | `-type f` 不匹配软链接，导致待处理数显示成 17 而非 42（仅影响分母显示） |

原脚本 `/root/autodl-tmp/asr/asr_run_and_verify.sh` **未改动**。

## 两个已排查的坑（结论）

### 1. mutagen 读不到 wav 时长 → 原脚本验收数字全是假的

`mutagen.File(x.wav).info.length` 对本批**所有** wav（原始的和 ffmpeg 抽的）都返回 `None`，
导致 `coverage = (last-first)/audio_dur` 里 `audio_dur=0`，覆盖率恒为 0%，
所有文件都会被判 `⚠️/❌` 且不移入 done。

历史日志 `/root/autodl-tmp/asr_process.log` 里那批"覆盖0%"是同一个 bug，不是转写质量问题。

**对策**：`verify.py` 改用 `ffprobe` 取时长（已验证对全部 42 个文件有效）。
主任务照跑不误——这个 bug 只影响验收判定，不影响 SRT 产出。

### 2. "开头缺 20 分钟"是真实静音，不是模型丢内容

`14、第四天千千老师` 首条字幕在 20:27。实测音量：

| 时间段 | mean | max |
|---|---|---|
| 0–1200s | −63 ~ −65 dB | −45 ~ −49 dB |
| 1250s 起 | −40 dB | −16.7 dB |

前 20 分半是开课前空等待（只有底噪），首条字幕内容正是
"各位组长可以提醒一下组内还没到的成员，马上开始上课了"。
尾部 max=7189.5s vs 音频 7189s，**零截断**。

**对策**：`verify.py` 对每个缺口段跑 `ffmpeg volumedetect` 实测，
`max_volume > -35dB` 才算"有语音却没转出文字"判 FAIL，静音段判通过。

## 验收命令

```bash
export PATH=/root/miniconda3/bin:$PATH
python /root/autodl-tmp/qc/verify.py /root/autodl-fs/qianchuan_asr \
       /root/autodl-tmp/qc/input /root/autodl-tmp/qc/done
```

输出每个文件的：音频时长 / 字幕结束点 / 条数 / 头缺 / 尾缺 / 中间最大空洞 / 判定。
**核心指标是"尾缺"**——模型注意力衰减会表现为提前收尾，这是唯一会静默丢内容的失败模式。

## 进度查看

```bash
grep -cE '✅ SRT saved' /root/autodl-tmp/qc/logs/asr_main.log   # 已完成数
grep -E '^\[[0-9]+/42\]' /root/autodl-tmp/qc/logs/asr_main.log | tail -3
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ls /root/autodl-fs/qianchuan_asr/*.srt | wc -l
```

## 断点续跑

脚本按"输出目录已有同名 .srt 就跳过"续跑，直接重跑即可，不会重复转写：

```bash
screen -dmS qcasr bash -c 'bash /root/autodl-tmp/qc/run_asr.sh course \
  /root/autodl-tmp/qc/input /root/autodl-fs/qianchuan_asr \
  >> /root/autodl-tmp/qc/logs/asr_main.log 2>&1'
```

## 回滚

```bash
screen -S qcasr -X quit; pkill -f run_asr.sh   # 停任务
rm -rf /root/autodl-tmp/qc                      # 删工作区（源文件不受影响）
rm -rf /root/autodl-fs/qianchuan_asr            # 删产出
```

源目录 `/root/autodl-fs/qianchuan` 全程只读，42 个文件未被修改或移动。

## 速度基准

600s 样本 62 秒跑完（含 ~20s 模型加载）；实跑首个文件 7189s 音频约 8.5 分钟，**约 14 倍速**。
全量 67h51m 预计 **5–6 小时**（含 42 次模型重复加载，每次约 20 秒）。
