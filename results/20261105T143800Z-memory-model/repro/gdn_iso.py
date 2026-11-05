"""Memory of gdn_checkpoint vs gated_delta_ops alone, real GDN shapes (diagnostic)."""
import sys
sys.path.insert(0, "tools")
import mlx.core as mx
from mlx_lm.models import gated_delta
import gdn_checkpoint

T, mode, grad_inputs = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
B, HK, HV, D = 1, 16, 32, 128
mx.random.seed(0)
q = mx.random.normal((B, T, HK, D)).astype(mx.bfloat16); k = mx.random.normal((B, T, HK, D)).astype(mx.bfloat16)
q, k = gated_delta.normalize_qk(q, k, inv_scale=D**-0.5, eps=1e-6)
v = mx.random.normal((B, T, HV, D)).astype(mx.bfloat16)
g = mx.random.uniform(0.9, 1.0, (B, T, HV)); beta = mx.random.uniform(0.05, 0.95, (B, T, HV))
st = mx.zeros((B, HV, D, D), dtype=mx.float32); mx.eval(q, k, v, g, beta, st)
fn = gated_delta.gated_delta_ops if mode == "ops" else gdn_checkpoint.gated_delta_ops_checkpointed
def f(q, k, v, g, beta):
    y, s = fn(q, k, v, g, beta, st, None)
    return y.astype(mx.float32).sum() + s.sum()
args = (0, 1, 2, 3, 4) if grad_inputs else (3,)
mx.reset_peak_memory(); base = mx.get_active_memory()
val, grads = mx.value_and_grad(f, argnums=args)(q, k, v, g, beta); mx.eval(val, grads)
pk = mx.get_peak_memory() - base
print(f"T={T} {mode} grad_inputs={int(grad_inputs)}: {pk/2**20:.0f} MiB = {pk/2**20/T:.2f} MiB/token")
