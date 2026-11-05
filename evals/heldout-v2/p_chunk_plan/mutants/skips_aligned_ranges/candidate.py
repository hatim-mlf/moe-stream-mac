def plan_byte_chunks(total_bytes, chunk_bytes, alignment):
    values=(total_bytes,chunk_bytes,alignment)
    if any(isinstance(v,bool) or not isinstance(v,int) for v in values):
        raise TypeError("sizes must be integers")
    if total_bytes<0 or chunk_bytes<=0 or alignment<=0 or chunk_bytes%alignment:
        raise ValueError("invalid byte plan")
    return [(start,min(chunk_bytes,total_bytes-start)) for start in range(0,total_bytes,chunk_bytes+alignment)]
