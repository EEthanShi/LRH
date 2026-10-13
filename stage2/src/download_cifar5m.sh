#!/bin/bash
# Download CIFAR-5m (Nakkiran et al. 2021) parts 0-5 from the public gresearch bucket, verify md5.
set -u
cd ~/hrh_v2_stage2_20261008/data/cifar5m
declare -A MD5=([0]=0EAcjCr5beUHP70wXkCzCg== [1]=egr653IGpD9NkTTF8/kLzg== [2]=DMtrPcJQVKkb/DzkHVM5lw== [3]=tztwlRtvbRzundi0TB/mIA== [4]=+JOEKAPUa237HeBKcaZ1jw== [5]=Gq+nsuqJjhxmIF1adAxyug==)
ok=1
for p in 0 1 2 3 4 5; do
  f=part$p.npz
  if [ -f $f.ok ]; then continue; fi
  curl -s --retry 5 -C - -o $f https://storage.googleapis.com/gresearch/cifar5m/$f || { echo "curl failed $f"; ok=0; continue; }
  got=$(openssl dgst -md5 -binary $f | base64)
  if [ "$got" == "${MD5[$p]}" ]; then touch $f.ok; echo "$(date -u +%FT%TZ) ok $f"; else echo "md5 mismatch $f $got"; ok=0; fi
done
if [ $ok == 1 ]; then touch DOWNLOAD.done; else touch DOWNLOAD.failed; fi
