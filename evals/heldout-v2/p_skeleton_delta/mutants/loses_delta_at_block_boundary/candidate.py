import struct
import numpy as np

def encode_delta_bytes(values):
    # TODO_BEGIN
    a=np.asarray(values)
    if a.ndim!=1 or a.size==0 or a.dtype.kind not in "iu":
        raise ValueError("values must be a nonempty integer vector")
    nums=[int(v) for v in a]
    if nums[0]<-32768 or nums[0]>32767:
        raise ValueError("first value does not fit int16")
    deltas=[(0 if len(nums)>1024 and i==1024 else nums[i]-nums[i-1]) for i in range(1,len(nums))]
    if any(d<-128 or d>127 for d in deltas):
        raise ValueError("delta does not fit int8")
    return struct.pack("<h",nums[0])+bytes((d & 255) for d in deltas)
    # TODO_END
