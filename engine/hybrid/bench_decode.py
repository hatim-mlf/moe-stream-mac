"""Reply (decode) speed of the hybrid engine on the 122B with different GPU expert-cache sizes.

  python engine/hybrid/bench_decode.py <model dir> <cache GB> [tokens]
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from hybrid import HybridModel  # noqa: E402

model_dir, cache_gb = Path(sys.argv[1]).expanduser(), float(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
predict = not (len(sys.argv) > 4 and sys.argv[4] == "nopredict")
hm = HybridModel(model_dir, cache_gb=cache_gb, predict=predict)
prompt = hm.tokenizer.apply_chat_template([{"role": "user", "content": "Write a short story about a lighthouse keeper."}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
logits = hm.forward(hm.tokenizer.encode(prompt))
hm.stats.__init__(); t0 = time.perf_counter(); out = []
for _ in range(n):
    tok = int(mx.argmax(logits)); out.append(tok); logits = hm.forward([tok])
dt = (time.perf_counter() - t0) / n
pstats = hm.predictive.stats() if hm.predictive else {"hit_rate": 0.0}
print(f"predict {str(predict):5} | cache {cache_gb:4.1f} GB | prediction hits {pstats['hit_rate']:.0%}: reply {dt:.2f} s/token ({1 / dt:.2f} tok/s) | read {hm.stats.read_s / n:.2f} s/token, "
      f"{hm.stats.bytes / n / 1e9:.2f} GB/token | hit rate {hm.expert_cache.hit_rate():.0%} | cached {hm.expert_cache.bytes_used / 1e9:.1f} GB "
      f"| GPU peak {mx.get_peak_memory() / 1e9:.1f} GB")
print("text:", repr(hm.tokenizer.decode(out[:30])))
