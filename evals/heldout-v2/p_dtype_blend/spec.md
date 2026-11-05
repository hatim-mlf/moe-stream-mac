Target: candidate.py (Python 3.14; standard library and NumPy only)

Blend two real numeric inputs with an explicit float32 output contract.

Exact public signature:
```python
def weighted_blend32(a, b, weight): ...
```

Compute `a*weight + b*(1-weight)` using NumPy broadcasting. Inputs may be integer or floating real arrays; reject bool, complex, object, and string dtypes with TypeError. Weight values must be finite and in [0,1], else ValueError. If shapes do not broadcast, raise ValueError. The returned array must always have dtype float32, including when all inputs are float64 or integers. Inputs are not modified.

Return only the complete file in one fenced python block.
