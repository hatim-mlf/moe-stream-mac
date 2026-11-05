from collections.abc import Mapping
def merge_count_maps(left,right):
 if not isinstance(left,Mapping) or not isinstance(right,Mapping): raise ValueError()
 out=dict(left)
 for k,v in right.items():
  if not isinstance(k,str) or isinstance(v,bool) or not isinstance(v,int) or v<0: raise ValueError()
  out[k]=out.get(k,0)+v
 return out
