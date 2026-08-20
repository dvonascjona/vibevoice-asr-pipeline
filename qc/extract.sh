#!/bin/bash
# mp4 -> 16k mono wav。源文件只读，产物写入 qc/input。
export PATH=/root/miniconda3/bin:$PATH
SRC=/root/autodl-fs/qianchuan
DST=/root/autodl-tmp/qc/input
fail=0; ok=0; n=0
total=$(ls "$SRC"/*.mp4 2>/dev/null | wc -l)
echo "[EXTRACT_START] $(date +%s) total=$total"
for f in "$SRC"/*.mp4; do
  [ -f "$f" ] || continue
  n=$((n+1))
  b=$(basename "$f"); b="${b%.*}"
  out="$DST/$b.wav"
  if [ -f "$out" ]; then echo "[$n/$total] SKIP $b"; ok=$((ok+1)); continue; fi
  src_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  ffmpeg -v error -y -i "$f" -vn -ar 16000 -ac 1 -c:a pcm_s16le -f wav "$out.part" 2>"$DST/.err_$n"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$n/$total] FAIL $b rc=$rc :: $(head -c 300 "$DST/.err_$n")"
    rm -f "$out.part"; fail=$((fail+1)); continue
  fi
  out_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out.part" 2>/dev/null)
  # 时长校验：抽出的音频必须与源视频时长相差 <2s，否则视为抽取不完整
  diff=$(awk -v a="$src_dur" -v b="$out_dur" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.1f", d}')
  bad=$(awk -v d="$diff" 'BEGIN{print (d>2)?1:0}')
  if [ "$bad" = "1" ]; then
    echo "[$n/$total] FAIL $b 时长不符 src=${src_dur}s out=${out_dur}s diff=${diff}s"
    rm -f "$out.part"; fail=$((fail+1)); continue
  fi
  mv "$out.part" "$out"
  echo "[$n/$total] OK $b ${out_dur}s"
  ok=$((ok+1))
done
rm -f "$DST"/.err_*
echo "[EXTRACT_END] $(date +%s) ok=$ok fail=$fail total=$total"
[ $fail -eq 0 ] || exit 1
