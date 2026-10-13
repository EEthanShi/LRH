#!/bin/bash
# e0-extract runner (not frozen code; calls only frozen scripts). Full extraction of each encoder, then finalize.
cd ~/hrh_v2_stage1_20261008 || exit 9
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export HF_HOME=$HOME/hrh_v2_stage1_20261008/extract/e0_sentence/hf_home HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
L=extract/e0_sentence/logs
for enc in mpnet contriever qwen3emb06; do
  echo "$(date -u +%FT%TZ) START full $enc" >> $L/markers.log
  taskset -c 0-5 venv/bin/python -I extract/e0_sentence/e0_extract.py full $enc > $L/full_$enc.log 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) END full $enc rc=$rc" >> $L/markers.log
  [ $rc -eq 0 ] || exit $rc
done
echo "$(date -u +%FT%TZ) START finalize" >> $L/markers.log
taskset -c 0-5 venv/bin/python -I extract/e0_sentence/e0_extract.py finalize > $L/finalize.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) END finalize rc=$rc" >> $L/markers.log
echo "$(date -u +%FT%TZ) ALLDONE full" >> $L/markers.log
