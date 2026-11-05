"""Does mx.checkpoint cut backward memory of a per-step recurrence? (diagnostic)"""
import sys
import mlx.core as mx

H, D, T, CHUNK = 32, 128, 256, 16
mx.random.seed(0)
q = mx.random.normal((T, H, D)); k = mx.random.normal((T, H, D)) * 0.05
v = mx.random.normal((T, H, D)); g = mx.random.uniform(0.9, 1.0, (T, H))

def step(state, t):
    s = state * g[t][:, None, None]
    delta = v[t] - (s * k[t][:, None, :]).sum(-1)
    s = s + delta[:, :, None] * k[t][:, None, :]
    return s, (s * q[t][:, None, :]).sum(-1)

def run(x0, lo, hi):
    s, ys = x0, []
    for t in range(lo, hi):
        s, y = step(s, t); ys.append(y)
    return s, mx.stack(ys)

def loss(x0, mode):
    s, outs = x0, []
    if mode == "plain":
        s, y = run(s, 0, T); return y.sum()
    for lo in range(0, T, CHUNK):
        f = mx.checkpoint(lambda st, lo=lo: run(st, lo, lo + CHUNK))
        s, y = f(s); outs.append(y)
    return mx.concatenate(outs).sum()

mode = sys.argv[1]
x0 = mx.zeros((H, D, D)); mx.eval(q, k, v, g, x0)
mx.reset_peak_memory(); base = mx.get_active_memory()
val, grad = mx.value_and_grad(lambda x: loss(x, mode))(x0); mx.eval(val, grad)
print(f"{mode}: backward peak {(mx.get_peak_memory()-base)/2**20:.0f} MiB = {(mx.get_peak_memory()-base)/2**21/T:.2f} states/token")
