#!/bin/bash
# Variant: LoRA keys without linear_attn.* (GDN layers frozen, FROZEN_GDN_KERNEL), so more layers may fit at 14k.
# The preflight models only adapted-GDN sets, so it is off (MEM_PREFLIGHT=0); mem_guard (29.5 GiB) still stops the run.
cd /Users/hatim/moe-stream-mac
R=results/20261123T203000Z-kat-lora-feasibility
KAT=$HOME/models/KAT-Coder-V2.5-Dev-OptiQ-4bit
for L in 4 8 16; do
  d=$R/nogdn-L$L; mkdir -p $d
  awk '$1=="scale:"{print "  scale: 4.0";next} $1=="max_seq_length:"{print "max_seq_length: 14336";next} /linear_attn\./{next} {print}' docs/finetune/lora.yaml > $d/lora.yaml
  env MEM_PREFLIGHT=0 FROZEN_GDN_KERNEL=1 MLX_CACHE_LIMIT_GB=0 .venv/bin/python tools/ft_lora_lowmem.py --train --config $d/lora.yaml \
    --model $KAT --data data/finetune/kat-pi-feasibility --num-layers $L --iters 2 --val-batches 1 \
    --steps-per-eval 2 --save-every 2 --adapter-path $d/adapter > $d/train.log 2>&1
  rc=$?
  echo "nogdn L=$L rc=$rc $(grep -o 'mlx_peak_memory_gib=[0-9.]*' $d/train.log) $(grep -o 's_per_step=[0-9.]*' $d/train.log) $(grep -o 'Trainable parameters: [^)]*)' $d/train.log)"
  [ $rc -eq 0 ] || break
done
echo NOGDN-DONE
