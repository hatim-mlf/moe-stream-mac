def merge_count_maps(left,right):
 out=dict(left)
 for k,v in right.items(): out[k]=out.get(k,0)+v
 return out
