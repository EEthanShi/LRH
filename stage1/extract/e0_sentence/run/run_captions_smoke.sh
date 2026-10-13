#!/bin/bash
# e0-extract runner (not frozen code; calls only frozen scripts). Captions, then the timing smoke of each encoder.
cd ~/hrh_v2_stage1_20261008 || exit 9
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export HF_HOME=$HOME/hrh_v2_stage1_20261008/extract/e0_sentence/hf_home HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
L=extract/e0_sentence/logs
echo "$(date -u +%FT%TZ) START captions" >> $L/markers.log
taskset -c 0-5 venv/bin/python -I extract/e0_sentence/e0_captions.py > $L/captions.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) END captions rc=$rc" >> $L/markers.log
[ $rc -eq 0 ] || exit $rc
for enc in mpnet contriever qwen3emb06; do
  echo "$(date -u +%FT%TZ) START smoke $enc" >> $L/markers.log
  taskset -c 0-5 venv/bin/python -I extract/e0_sentence/e0_extract.py smoke $enc > $L/smoke_$enc.log 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) END smoke $enc rc=$rc" >> $L/markers.log
  [ $rc -eq 0 ] || exit $rc
done
echo "$(date -u +%FT%TZ) ALLDONE captions_smoke" >> $L/markers.log
