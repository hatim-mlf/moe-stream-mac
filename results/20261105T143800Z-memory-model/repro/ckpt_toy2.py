"""Toy recurrence: does checkpoint still bound memory when per-step inputs need gradients? (diagnostic)"""
import sys
import mlx.core as mx

T, mode, CHUNK = int(sys.argv[1]), sys.argv[2], 32
H, D = 32, 128
mx.random.seed(0)
q = mx.random.normal((T, H, D)); k = mx.random.normal((T, H, D)) * 0.05; v = mx.random.normal((T, H, D))
x0 = mx.zeros((H, D, D)); mx.eval(q, k, v, x0)

def run(s, qc, kc, vc):
    ys = []
    for t in range(qc.shape[0]):
        s = s * 0.95
        delta = vc[t] - (s * kc[t][:, None, :]).sum(-1)
        s = s + delta[:, :, None] * kc[t][:, None, :]
        ys.append((s * qc[t][:, None, :]).sum(-1))
    return s, mx.stack(ys)

def loss(q, k, v):
    if mode == "plain":
        return run(x0, q, k, v)[1].sum()
    s, outs = x0, []
    for lo in range(0, T, CHUNK):
        s, y = mx.checkpoint(run)(s, q[lo:lo+CHUNK], k[lo:lo+CHUNK], v[lo:lo+CHUNK]); outs.append(y)
    return mx.concatenate(outs).sum()

mx.reset_peak_memory(); base = mx.get_active_memory()
val, grads = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v); mx.eval(val, grads)
pk = mx.get_peak_memory() - base
print(f"T={T} {mode} (grads wrt q,k,v): {pk/2**20:.0f} MiB = {pk/2**21/T:.2f} states/token")
