#!/bin/bash
# Brief 52 Phase B step 4 feasibility: does a LoRA step on KAT (OptiQ mixed 4/8-bit) fit with a 16k-token pi session?
# Two steps per depth, stop at the first failure. mem_guard (inside ft_lora_lowmem.py) kills the run before a wired-memory panic.
cd /Users/hatim/moe-stream-mac
R=results/20261123T203000Z-kat-lora-feasibility
KAT=$HOME/models/KAT-Coder-V2.5-Dev-OptiQ-4bit
for L in 1 4 8 16; do
  mkdir -p $R/L$L
  awk '$1=="scale:"{print "  scale: 4.0";next} $1=="max_seq_length:"{print "max_seq_length: 14336";next} {print}' docs/finetune/lora.yaml > $R/L$L/lora.yaml
  env FROZEN_GDN_KERNEL=1 MLX_CACHE_LIMIT_GB=0 .venv/bin/python tools/ft_lora_lowmem.py --train --config $R/L$L/lora.yaml \
    --model $KAT --data data/finetune/kat-pi-feasibility --num-layers $L --iters 2 --val-batches 1 \
    --steps-per-eval 2 --save-every 2 --adapter-path $R/L$L/adapter > $R/L$L/train.log 2>&1
  rc=$?
  echo "L=$L rc=$rc $(grep -o 'mlx_peak_memory_gib=[0-9.]*' $R/L$L/train.log) $(grep -o 's_per_step=[0-9.]*' $R/L$L/train.log) $(grep -oE 'Val loss [0-9.]+' $R/L$L/train.log | tail -1)"
  [ $rc -eq 0 ] || break
done
echo FEAS-DONE
