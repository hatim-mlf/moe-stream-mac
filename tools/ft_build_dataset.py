#!/usr/bin/env python3
"""Build a task-disjoint MLX chat SFT set plus preference/review sidecars."""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/finetune"
SYSTEM = ("You are a senior software engineer (C and Python). Follow the brief exactly. "
          "Reply with ONE fenced code block containing the requested code, and nothing else.\n\n" +
          (ROOT / "tools/helper_rules.md").read_text())
HELDOUT = {"qmat", "ops", "gdn", "qmat-row", "qgroup", "expert-cache", "predictive", "pinned"}
TRAIN_IDS = {"gdn-layer", "attn-layer", "moe-layer", "qchat", "fetch-experts", "read-sorted"}
VALID_IDS = {"model-forward", "expert-reader"}
NATURAL = [
    {"id": "gdn-layer", "label": "gdn-layer", "path": "engine/src/gdn_layer.c", "rev": "4acd7d7", "selector": None},
    {"id": "attn-layer", "label": "attn", "path": "engine/src/attn_layer.c", "rev": "dd0430b", "selector": None},
    {"id": "moe-layer", "label": "moe", "path": "engine/src/moe_layer.c", "rev": "e98c2f7", "selector": None},
    {"id": "model-forward", "label": "model-fwd", "path": "engine/src/model.c", "rev": "1723786", "selector": None},
    {"id": "qchat", "label": "qchat", "path": "engine/cli/qchat.py", "rev": "93d0071", "selector": "python:qchat"},
    {"id": "fetch-experts", "label": "fetch", "path": "engine/src/moe_layer.c", "rev": "5b2ec01", "selector": "c:fetch_experts"},
    {"id": "expert-reader", "label": "expert-reader", "path": "engine/hybrid/expert_reader.py", "rev": "144ea14", "selector": None},
    {"id": "read-sorted", "label": "read-sorted", "path": "engine/hybrid/expert_reader.py", "rev": "9a6af1e", "selector": "python:read_sorted"},
]
SEEDS = [
    {"id":"seed-c-u4", "kind":"c99", "brief":"Write `uint8_t unpack_u4(const uint8_t *p, size_t i)`. Return packed nibble i, low nibble first. Assume i is valid.", "code":"#include <stdint.h>\n#include <stddef.h>\nuint8_t unpack_u4(const uint8_t *p, size_t i) { uint8_t b = p[i / 2]; return (uint8_t)((b >> ((i % 2) * 4)) & 15u); }", "tests":"#include <assert.h>\n#include <stdint.h>\n#include \"candidate.c\"\nint main(void) { uint8_t p[]={0xA3,0xF0}; assert(unpack_u4(p,0)==3); assert(unpack_u4(p,1)==10); assert(unpack_u4(p,2)==0); assert(unpack_u4(p,3)==15); }"},
    {"id":"seed-c-dot-u8", "kind":"c99", "brief":"Implement `float dot_u8(const uint8_t *q, const float *x, size_t n)`: sum unsigned q[i]*x[i] in float, with no allocation.", "code":"#include <stdint.h>\n#include <stddef.h>\nfloat dot_u8(const uint8_t *q, const float *x, size_t n) { float s=0.0f; for (size_t i=0;i<n;i++) s += (float)q[i] * x[i]; return s; }", "tests":"#include <assert.h>\n#include <stdint.h>\n#include \"candidate.c\"\nint main(void) { uint8_t q[]={0,255,2}; float x[]={7,-1,0.5f}; assert(dot_u8(q,x,3)==-254.0f); assert(dot_u8(q,x,0)==0.0f); }"},
    {"id":"seed-c-bf16", "kind":"c99", "brief":"Implement `float bf16_to_float(uint16_t h)` by placing the bf16 bits in the high half of float32 and using memcpy, never pointer casts.", "code":"#include <stdint.h>\n#include <string.h>\nfloat bf16_to_float(uint16_t h) { uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,sizeof f); return f; }", "tests":"#include <assert.h>\n#include <stdint.h>\n#include <string.h>\n#include \"candidate.c\"\nint main(void) { assert(bf16_to_float(0x3f80)==1.0f); assert(bf16_to_float(0xc020)==-2.5f); }"},
    {"id":"seed-c-sigmoid", "kind":"c99", "brief":"Write a numerically stable `float stable_sigmoid(float x)` that avoids overflow for large positive and negative inputs.", "code":"#include <math.h>\nfloat stable_sigmoid(float x) { if (x >= 0.0f) { float e=expf(-x); return 1.0f/(1.0f+e); } float e=expf(x); return e/(1.0f+e); }", "tests":"#include <assert.h>\n#include <math.h>\n#include \"candidate.c\"\nint main(void) { assert(fabsf(stable_sigmoid(0)-0.5f)<1e-6f); assert(stable_sigmoid(1000)>0.999f); assert(stable_sigmoid(-1000)<0.001f); }"},
    {"id":"seed-c-clamp", "kind":"c99", "brief":"Implement `int clamp_int(int x, int low, int high)`, returning low/high at the bounds and x inside. Assume low <= high.", "code":"int clamp_int(int x, int low, int high) { return x < low ? low : x > high ? high : x; }", "tests":"#include <assert.h>\n#include \"candidate.c\"\nint main(void) { assert(clamp_int(-4,-2,5)==-2); assert(clamp_int(3,-2,5)==3); assert(clamp_int(8,-2,5)==5); }"},
    {"id":"seed-c-satadd", "kind":"c99", "brief":"Implement saturating `int32_t sat_add_i32(int32_t a, int32_t b)` without signed overflow.", "code":"#include <stdint.h>\n#include <limits.h>\nint32_t sat_add_i32(int32_t a, int32_t b) { if (b>0 && a>INT32_MAX-b) return INT32_MAX; if (b<0 && a<INT32_MIN-b) return INT32_MIN; return a+b; }", "tests":"#include <assert.h>\n#include <stdint.h>\n#include <limits.h>\n#include \"candidate.c\"\nint main(void) { assert(sat_add_i32(3,4)==7); assert(sat_add_i32(INT32_MAX,1)==INT32_MAX); assert(sat_add_i32(INT32_MIN,-1)==INT32_MIN); }"},
    {"id":"seed-c-topk", "kind":"skeleton-c", "brief":"Fill the TODO in `void topk_indices(const float *x,int n,int k,int *idx)`. Output indices of k largest values, descending; equal values keep lower index first. n<=64, 1<=k<=n, no allocation. Return the complete function.", "code":"void topk_indices(const float *x,int n,int k,int *idx) { int count=0; for(int i=0;i<n;i++){ int p=0; while(p<count && x[idx[p]]>=x[i]) p++; if(p>=k) continue; int end=count<k?count:k-1; for(int j=end;j>p;j--) idx[j]=idx[j-1]; idx[p]=i; if(count<k) count++; } }", "tests":"#include <assert.h>\n#include \"candidate.c\"\nint main(void) { float x[]={3,9,9,-1,4}; int i[3]={0}; topk_indices(x,5,3,i); assert(i[0]==1 && i[1]==2 && i[2]==4); }"},
    {"id":"seed-c-ring", "kind":"skeleton-c", "brief":"Fill `int ring_push(int *buf,int cap,int *head,int *size,int value)`. Store at the logical tail, increment size, return 0; invalid/full input returns -1 without mutation. head is oldest item.", "code":"int ring_push(int *buf,int cap,int *head,int *size,int value) { if(!buf || !head || !size || cap<=0 || *head<0 || *head>=cap || *size<0 || *size>=cap) return -1; buf[(*head+*size)%cap]=value; (*size)++; return 0; }", "tests":"#include <assert.h>\n#include \"candidate.c\"\nint main(void) { int b[2]={0,0},h=1,n=0; assert(ring_push(b,2,&h,&n,7)==0 && n==1 && b[1]==7); assert(ring_push(b,2,&h,&n,8)==0 && n==2 && b[0]==8); assert(ring_push(b,2,&h,&n,9)==-1 && n==2 && b[0]==8); }"},
    {"id":"seed-c-strided", "kind":"c99", "brief":"Implement `float sum_strided(const float *x,size_t n,size_t stride)`, summing x[0], x[stride], ... x[(n-1)*stride]. No allocation.", "code":"#include <stddef.h>\nfloat sum_strided(const float *x,size_t n,size_t stride) { float s=0.0f; for(size_t i=0;i<n;i++) s+=x[i*stride]; return s; }", "tests":"#include <assert.h>\n#include \"candidate.c\"\nint main(void) { float x[]={1,99,2,99,3}; assert(sum_strided(x,3,2)==6); assert(sum_strided(x,0,2)==0); }"},
    {"id":"seed-c-argmax", "kind":"c99", "brief":"Implement `int argmax_first(const float *x,int n)`: return the first index of the maximum, or -1 if n<=0.", "code":"int argmax_first(const float *x,int n) { if(n<=0) return -1; int best=0; for(int i=1;i<n;i++) if(x[i]>x[best]) best=i; return best; }", "tests":"#include <assert.h>\n#include \"candidate.c\"\nint main(void) { float x[]={-2,4,4,1}; assert(argmax_first(x,4)==1); assert(argmax_first(x,0)==-1); }"},
    {"id":"seed-py-chunks", "kind":"python", "brief":"Write `chunks(values, size)` returning consecutive list chunks. Raise ValueError for size<1; empty input returns [].", "code":"def chunks(values, size):\n    if size < 1:\n        raise ValueError('size must be positive')\n    return [values[i:i + size] for i in range(0, len(values), size)]", "tests":"assert chunks([1,2,3,4,5],2)==[[1,2],[3,4],[5]]\nassert chunks([],3)==[]\ntry: chunks([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-py-unique", "kind":"python", "brief":"Write `stable_unique(values)` returning the first occurrence of each hashable value in input order.", "code":"def stable_unique(values):\n    seen=set()\n    out=[]\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            out.append(value)\n    return out", "tests":"assert stable_unique([3,1,3,2,1])==[3,1,2]\nassert stable_unique([])==[]\nassert stable_unique(['a','a','b'])==['a','b']"},
    {"id":"seed-py-intervals", "kind":"python", "brief":"Write `merge_intervals(intervals)`: sort integer (start,end) pairs and merge overlapping or touching intervals. Return new pairs.", "code":"def merge_intervals(intervals):\n    out=[]\n    for start,end in sorted(intervals):\n        if out and start <= out[-1][1]:\n            out[-1][1]=max(out[-1][1],end)\n        else:\n            out.append([start,end])\n    return [tuple(pair) for pair in out]", "tests":"assert merge_intervals([(5,7),(1,3),(3,4),(10,10)])==[(1,4),(5,7),(10,10)]\nassert merge_intervals([])==[]\nassert merge_intervals([(2,8),(3,4)])==[(2,8)]"},
    {"id":"seed-py-rotate", "kind":"python", "brief":"Write `rotate_right(values, k)` returning a right-rotated copy. Support negative and oversized k; empty input returns [].", "code":"def rotate_right(values, k):\n    if not values:\n        return []\n    k %= len(values)\n    return values[-k:] + values[:-k] if k else list(values)", "tests":"assert rotate_right([1,2,3,4],1)==[4,1,2,3]\nassert rotate_right([1,2,3],-1)==[2,3,1]\nassert rotate_right([1],100)==[1]\nassert rotate_right([],2)==[]"},
    {"id":"seed-py-runs", "kind":"python", "brief":"Write `group_consecutive(values)` for strictly increasing integers, returning maximal consecutive runs as lists.", "code":"def group_consecutive(values):\n    if not values:\n        return []\n    groups=[[values[0]]]\n    for value in values[1:]:\n        if value==groups[-1][-1]+1:\n            groups[-1].append(value)\n        else:\n            groups.append([value])\n    return groups", "tests":"assert group_consecutive([1,2,3,7,9,10])==[[1,2,3],[7],[9,10]]\nassert group_consecutive([])==[]\nassert group_consecutive([-2,-1,2])==[[-2,-1],[2]]"},
    {"id":"seed-py-kv", "kind":"python", "brief":"Write `parse_key_values(text, sep='=')` parsing nonempty lines at the first separator, trimming key/value whitespace. Ignore blank lines; raise ValueError on a nonblank malformed line.", "code":"def parse_key_values(text, sep='='):\n    result={}\n    for line in text.splitlines():\n        if not line.strip():\n            continue\n        if sep not in line:\n            raise ValueError('missing separator')\n        key,value=line.split(sep,1)\n        result[key.strip()]=value.strip()\n    return result", "tests":"assert parse_key_values(' a = x=y\\n\\n b = z\\n')=={'a':'x=y','b':'z'}\nassert parse_key_values('k:v', ':')=={'k':'v'}\ntry: parse_key_values('bad')\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-py-window", "kind":"skeleton-python", "brief":"Fill the TODO in `sliding_sums(values, window)`: return every contiguous window sum; raise ValueError for window<=0; return [] when window>len(values).", "code":"def sliding_sums(values, window):\n    if window <= 0:\n        raise ValueError('window must be positive')\n    if window > len(values):\n        return []\n    total=sum(values[:window])\n    out=[total]\n    for i in range(window,len(values)):\n        total += values[i]-values[i-window]\n        out.append(total)\n    return out", "tests":"assert sliding_sums([1,2,3,4],2)==[3,5,7]\nassert sliding_sums([],1)==[]\nassert sliding_sums([1],2)==[]\ntry: sliding_sums([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-py-flatten", "kind":"skeleton-python", "brief":"Fill `flatten_one_level(rows)` to concatenate one level of nested lists without recursively flattening values.", "code":"def flatten_one_level(rows):\n    return [value for row in rows for value in row]", "tests":"assert flatten_one_level([[1,2],[],[3]])==[1,2,3]\nassert flatten_one_level([])==[]\nassert flatten_one_level([[1],[2,3]])==[1,2,3]"},
    {"id":"seed-py-transpose", "kind":"python", "brief":"Write `transpose_rect(rows)` returning columns as lists. Empty input returns []; ragged rows raise ValueError.", "code":"def transpose_rect(rows):\n    if not rows:\n        return []\n    width=len(rows[0])\n    if any(len(row)!=width for row in rows):\n        raise ValueError('ragged rows')\n    return [[row[i] for row in rows] for i in range(width)]", "tests":"assert transpose_rect([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]\nassert transpose_rect([[],[]])==[]\nassert transpose_rect([])==[]\ntry: transpose_rect([[1],[2,3]])\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-py-clamp", "kind":"python", "brief":"Write `clamp_all(values, low, high)` returning a new list with values clamped inclusively. Raise ValueError if low>high.", "code":"def clamp_all(values, low, high):\n    if low > high:\n        raise ValueError('low exceeds high')\n    return [min(high,max(low,value)) for value in values]", "tests":"assert clamp_all([-3,0,8],-1,5)==[-1,0,5]\nassert clamp_all([],0,1)==[]\ntry: clamp_all([1],2,1)\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-np-softmax", "kind":"numpy", "brief":"Implement `stable_softmax(x, axis=-1)` with NumPy, subtracting the maximum and supporting any valid axis.", "code":"import numpy as np\ndef stable_softmax(x, axis=-1):\n    a=np.asarray(x,dtype=np.float64)\n    shifted=a-np.max(a,axis=axis,keepdims=True)\n    e=np.exp(shifted)\n    return e/np.sum(e,axis=axis,keepdims=True)", "tests":"import numpy as np\ny=stable_softmax(np.array([[1000.,1001.],[-2.,-2.]]))\nassert np.allclose(y.sum(axis=1),1) and np.all(np.isfinite(y))\nassert np.allclose(y[1],[0.5,0.5])"},
    {"id":"seed-np-row-norm", "kind":"numpy", "brief":"Implement `row_l2_normalize(x, eps=1e-6)` returning rows divided by max(norm, eps); zero rows stay zero.", "code":"import numpy as np\ndef row_l2_normalize(x, eps=1e-6):\n    a=np.asarray(x,dtype=np.float64)\n    norms=np.sqrt(np.sum(a*a,axis=-1,keepdims=True))\n    return a/np.maximum(norms,eps)", "tests":"import numpy as np\nx=np.array([[3.,4.],[0.,0.]])\ny=row_l2_normalize(x)\nassert np.allclose(y[0],[0.6,0.8]) and np.array_equal(y[1],[0.,0.])"},
    {"id":"seed-np-topk", "kind":"numpy", "brief":"Implement `topk_stable(x,k)` returning indices of k largest 1-D values, descending with lower index first on ties. Validate 1<=k<=len(x).", "code":"import numpy as np\ndef topk_stable(x,k):\n    a=np.asarray(x)\n    if a.ndim!=1 or k<1 or k>a.size:\n        raise ValueError('invalid k or input')\n    return np.lexsort((np.arange(a.size),-a))[:k]", "tests":"import numpy as np\nassert np.array_equal(topk_stable(np.array([2,5,5,1]),3),[1,2,0])\ntry: topk_stable(np.array([1]),2)\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-np-affine", "kind":"numpy", "brief":"Implement `affine_dequant(q, scale, bias)` as unsigned affine dequantization q*scale+bias with NumPy broadcasting.", "code":"import numpy as np\ndef affine_dequant(q,scale,bias):\n    return np.asarray(q,dtype=np.uint8)*scale+bias", "tests":"import numpy as np\ngot=affine_dequant(np.array([0,15],dtype=np.uint8),np.array(2.),np.array(-1.))\nassert np.allclose(got,[-1.,29.])"},
    {"id":"seed-np-column-sum", "kind":"skeleton-numpy", "brief":"Fill `column_sums(x)` to return float64 column totals for a 2-D NumPy array; reject other ranks.", "code":"import numpy as np\ndef column_sums(x):\n    a=np.asarray(x)\n    if a.ndim!=2:\n        raise ValueError('expected a matrix')\n    return np.sum(a,axis=0,dtype=np.float64)", "tests":"import numpy as np\nassert np.allclose(column_sums(np.array([[1,2],[3,4]])),[4,6])\ntry: column_sums(np.array([1,2]))\nexcept ValueError: pass\nelse: raise AssertionError('expected ValueError')"},
    {"id":"seed-np-mask", "kind":"numpy", "brief":"Implement `mask_values(x, low, high)` returning a boolean mask for inclusive bounds low<=x<=high; reject low>high.", "code":"import numpy as np\ndef mask_values(x,low,high):\n    if low>high:\n        raise ValueError('low exceeds high')\n    a=np.asarray(x)\n    return (a>=low)&(a<=high)", "tests":"import numpy as np\nassert np.array_equal(mask_values(np.array([-1,0,2,4]),0,2),[False,True,True,False])"},
    {"id":"seed-py-partition", "kind":"python", "brief":"Write `partition_by(values, predicate)` returning two lists: values where predicate is true, then false, preserving order.", "code":"def partition_by(values,predicate):\n    yes=[]\n    no=[]\n    for value in values:\n        (yes if predicate(value) else no).append(value)\n    return yes,no", "tests":"assert partition_by([1,2,3,4],lambda x:x%2==0)==([2,4],[1,3])\nassert partition_by([],bool)==([],[])"},
    {"id":"seed-py-split-lines", "kind":"python", "brief":"Write `split_nonempty_lines(text)` returning stripped nonblank lines in order.", "code":"def split_nonempty_lines(text):\n    return [line.strip() for line in text.splitlines() if line.strip()]", "tests":"assert split_nonempty_lines(' a \\n\\n b\\n')==['a','b']\nassert split_nonempty_lines('')==[]"},
]


def git_text(revision, path):
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, check=True).stdout


def extract_python(source, names):
    tree = ast.parse(source)
    lines = source.splitlines()
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            nodes.update({child.name: child for child in node.body
                          if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))})
    return "\n\n".join("\n".join(lines[nodes[name].lineno - 1:nodes[name].end_lineno]) for name in names)


def extract_c_function(source, name):
    start = re.search(r"(?m)^static\s+\w[\w\s*]*\b" + re.escape(name) + r"\s*\(", source)
    if not start:
        raise ValueError(f"missing C function {name}")
    brace = source.index("{", start.start())
    depth = 0
    for index in range(brace, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start.start():index + 1]
    raise ValueError(f"unclosed C function {name}")


def accepted_code(task):
    code = git_text(task["rev"], task["path"])
    if task["selector"] == "python:qchat":
        return extract_python(code, ["sample", "Printer", "footer", "handle_command"])
    if task["selector"] == "python:read_sorted":
        return extract_python(code, ["num_experts", "read_sorted"])
    if task["selector"] == "c:fetch_experts":
        return extract_c_function(code, "fetch_experts")
    return code


def code_fence(code, language):
    return f"```{language}\n{code.strip()}\n```"


def raw_attempts(label):
    return sorted((ROOT / "results/delegations").glob(f"*-{label}-try*"))


def first_code(label):
    attempts = raw_attempts(label)
    if not attempts:
        return ""
    reply = (attempts[0] / "reply.md").read_text(errors="replace")
    match = re.search(r"^```[A-Za-z0-9_+-]*\n(.*?)\n```[ \t]*$", reply, re.S | re.M)
    return match.group(1) if match else ""


def make_messages(prompt, code):
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": code}]}


def validate_solution(kind, code, tests, timeout=20):
    with tempfile.TemporaryDirectory(prefix="ft-seed-") as folder:
        root = Path(folder)
        if kind in {"c99", "skeleton-c"}:
            (root / "candidate.c").write_text(code)
            (root / "test.c").write_text(tests)
            binary = root / "check"
            try:
                compile_result = subprocess.run(["cc", "-std=c99", "-O0", "-Wall", "-Wextra", "-Werror",
                                                 str(root / "test.c"), "-lm", "-o", str(binary)],
                                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False, "C compilation timed out"
            if compile_result.returncode:
                return False, compile_result.stdout
            command = [str(binary)]
        else:
            (root / "check.py").write_text(code + "\n\n" + tests + "\n")
            command = [str(ROOT / ".venv/bin/python"), str(root / "check.py")]
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "candidate tests timed out"
        return result.returncode == 0, result.stdout


def teacher_candidate(endpoint, model, category):
    prompt = (f"Create one new {category} coding task matching small repository delegation briefs. "
              "Return JSON only with keys language (c99 or python), prompt, solution, tests. "
              "For C, tests must include candidate.c and a main with assertions. For Python, tests must be "
              "assertions appended after solution. Cover edge cases; use no network or external packages beyond numpy.")
    body = {"model": model, "temperature": 0.8, "max_tokens": 5000,
            "messages": [{"role": "system", "content": "Produce a novel task and a correct reference solution."},
                         {"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("FT_TEACHER_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        text = json.load(response)["choices"][0]["message"]["content"]
    candidate = json.loads(text)
    if candidate.get("language") not in {"c99", "python"}:
        raise ValueError("teacher language must be c99 or python")
    return candidate


def natural_examples():
    train, valid, preferences, reviews, meta = [], [], [], [], []
    for task in NATURAL:
        prompt_file = next((ROOT / "results/delegations").glob(f"*-{task['label']}-try1/prompt.md"))
        prompt = prompt_file.read_text()
        accepted = accepted_code(task)
        code = code_fence(accepted, "python" if task["path"].endswith(".py") else "c")
        row = make_messages(prompt, code)
        row["task_id"] = task["id"]
        bucket = valid if task["id"] in VALID_IDS else train
        bucket.append(row)
        attempts = raw_attempts(task["label"])
        rejected = first_code(task["label"])
        failed_once = task["id"] == "fetch-experts"
        if task["id"] in TRAIN_IDS and rejected and (len(attempts) > 1 or failed_once):
            preferences.append({"task_id": task["id"], "prompt": row["messages"][:-1],
                                "chosen": [row["messages"][-1]],
                                "rejected": [{"role": "assistant", "content": code_fence(
                                    rejected, "python" if task["path"].endswith(".py") else "c")} ]})
        if task["id"] in TRAIN_IDS | VALID_IDS and len(attempts) > 1:
            latest_prompt = (attempts[-1] / "prompt.md").read_text(errors="replace")
            reviews.append(make_messages(latest_prompt, row["messages"][-1]["content"]) | {"task_id": task["id"]})
        meta.append({"task_id": task["id"], "split": "valid" if task["id"] in VALID_IDS else "train",
                     "revision": task["rev"], "path": task["path"],
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "accepted_sha256": hashlib.sha256(accepted.encode()).hexdigest()})
    return train, valid, preferences, reviews, meta


def seed_examples():
    train, valid, rows = [], [], []
    for index, task in enumerate(SEEDS):
        passed, output = validate_solution(task["kind"], task["code"], task["tests"])
        if not passed:
            raise RuntimeError(f"seed rejected: {task['id']}: {output}")
        row = make_messages(task["brief"], code_fence(task["code"], "c" if task["kind"] in {"c99", "skeleton-c"} else "python"))
        row["task_id"] = task["id"]
        bucket = valid if index % 5 == 0 else train
        bucket.append(row)
        rows.append({"task_id": task["id"], "split": "valid" if index % 5 == 0 else "train",
                     "kind": task["kind"], "passed": True})
    return train, valid, rows


def teacher_examples(count, endpoint, model, max_rejections):
    accepted, rejected = [], []
    categories = ("C99 kernel with fixture tests", "Python standard-library utility with unit tests",
                  "Python NumPy utility with unit tests", "skeleton-fill task in C99 or Python")
    for index in range(count):
        for trial in range(max_rejections):
            try:
                item = teacher_candidate(endpoint, model, categories[index % len(categories)])
                if not all(isinstance(item.get(key), str) and item[key].strip()
                           for key in ("prompt", "solution", "tests")):
                    raise ValueError("teacher response needs nonempty prompt, solution, and tests")
                ok, output = validate_solution(item["language"], item["solution"], item["tests"])
            except (KeyError, TypeError, ValueError) as error:
                item, ok, output = {"error": str(error)}, False, str(error)
            if ok:
                item["task_id"] = f"teacher-{index:04d}"
                item["split"] = "valid" if index % 5 == 0 else "train"
                accepted.append(item)
                break
            rejected.append({"index": index, "trial": trial + 1, "item": item, "failure": output[-2000:]})
        else:
            raise RuntimeError(f"teacher rejection limit reached at item {index}")
    return accepted, rejected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def verify_no_heldout(rows):
    leaking = sorted({row.get("task_id") for row in rows} & HELDOUT)
    if leaking:
        raise RuntimeError("held-out task data found: " + ", ".join(leaking))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--teacher-url", help="OpenAI-compatible /chat/completions endpoint for a stronger teacher")
    p.add_argument("--teacher-model")
    p.add_argument("--teacher-count", type=int, default=0)
    p.add_argument("--max-rejections", type=int, default=5)
    return p.parse_args(argv)


def assemble_rows(args):
    natural_train, natural_valid, prefs, reviews, natural_meta = natural_examples()
    seed_only_train, seed_only_valid, seed_meta = seed_examples()
    seed_train, seed_valid = list(seed_only_train), list(seed_only_valid)
    teachers, rejected = teacher_examples(args.teacher_count, args.teacher_url,
                                           args.teacher_model, args.max_rejections) if args.teacher_count else ([], [])
    for item in teachers:
        row = make_messages(item["prompt"], code_fence(item["solution"], "c" if item["language"] == "c99" else "python"))
        row["task_id"] = item["task_id"]
        (seed_valid if item["split"] == "valid" else seed_train).append(row)
    review_train = [r for r in reviews if r["task_id"] in TRAIN_IDS]
    review_valid = [r for r in reviews if r["task_id"] in VALID_IDS]
    train = natural_train + seed_train + review_train
    valid = natural_valid + seed_valid + review_valid
    long_train = next(row for row in review_train if row["task_id"] == "attn-layer")
    long_valid = next(row for row in review_valid if row["task_id"] == "model-forward")
    probe_train, probe_valid = seed_only_train[:4] + [long_train], [long_valid]
    verify_no_heldout(train + valid + prefs + reviews + probe_train + probe_valid)
    return {"natural_meta": natural_meta, "seed_meta": seed_meta, "seed_only_train": seed_only_train,
            "seed_only_valid": seed_only_valid, "teachers": teachers, "rejected": rejected,
            "probe_train": probe_train, "probe_valid": probe_valid, "train": train, "valid": valid,
            "prefs": prefs, "reviews": reviews, "review_train": review_train,
            "review_valid": review_valid}


def dataset_manifest(data):
    return {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                        stdout=subprocess.PIPE, check=True).stdout.strip(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "heldout_task_ids": sorted(HELDOUT), "train_eligible": sorted(TRAIN_IDS | VALID_IDS),
            "train_task_ids": sorted(TRAIN_IDS), "valid_task_ids": sorted(VALID_IDS),
            "natural": data["natural_meta"],
            "seed_validation": [r["task_id"] for r in data["seed_meta"] if r["split"] == "valid"],
            "seed_train": [r["task_id"] for r in data["seed_meta"] if r["split"] == "train"],
            "seed_validated": len(data["seed_meta"]),
            "seed_only_train_rows": len(data["seed_only_train"]),
            "seed_only_valid_rows": len(data["seed_only_valid"]),
            "teacher_accepted": len(data["teachers"]), "teacher_rejected": len(data["rejected"]),
            "probe_train_rows": len(data["probe_train"]), "probe_valid_rows": len(data["probe_valid"]),
            "probe_train_task_ids": [r["task_id"] for r in data["probe_train"]],
            "probe_valid_task_ids": [r["task_id"] for r in data["probe_valid"]],
            "train_rows": len(data["train"]), "valid_rows": len(data["valid"]),
            "preference_rows": len(data["prefs"]), "review_rows": len(data["reviews"]),
            "review_train_rows": len(data["review_train"]),
            "review_valid_rows": len(data["review_valid"])}


def write_outputs(args, data):
    for path, rows in (("seed_only/train.jsonl", data["seed_only_train"]),
                       ("seed_only/valid.jsonl", data["seed_only_valid"]),
                       ("probe/train.jsonl", data["probe_train"]), ("probe/valid.jsonl", data["probe_valid"]),
                       ("train.jsonl", data["train"]), ("valid.jsonl", data["valid"]),
                       ("preferences.jsonl", data["prefs"]), ("review_follow.jsonl", data["reviews"]),
                       ("rejected.jsonl", data["rejected"])):
        write_jsonl(args.out / path, rows)
    manifest = dataset_manifest(data)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("seed_validated", "teacher_accepted", "teacher_rejected",
                                                 "seed_only_train_rows", "seed_only_valid_rows",
                                                 "probe_train_rows", "probe_valid_rows",
                                                 "train_rows", "valid_rows", "preference_rows",
                                                 "review_train_rows", "review_valid_rows")}, indent=2))


def main(argv=None):
    args = arguments(argv)
    if args.teacher_count and not (args.teacher_url and args.teacher_model):
        raise SystemExit("--teacher-count requires --teacher-url and --teacher-model")
    write_outputs(args, assemble_rows(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
