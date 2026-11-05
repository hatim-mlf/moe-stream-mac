from collections.abc import Mapping

def merge_count_maps(left,right):
    if not isinstance(left,Mapping) or not isinstance(right,Mapping): raise ValueError("expected mappings")
    out={}
    for source in (left,right):
        for key,value in source.items():
            if not isinstance(key,str) or isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError("invalid count")
            out[key]=out.get(key,0)+value
    return out
