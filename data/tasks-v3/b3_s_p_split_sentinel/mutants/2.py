def split_on_sentinel(values,sentinel):
 out=[]; cur=[]
 for x in values:
  if x==sentinel:
   if cur: out.append(cur); cur=[]
  else: cur.append(x)
 if cur: out.append(cur)
 return out[::-1]
