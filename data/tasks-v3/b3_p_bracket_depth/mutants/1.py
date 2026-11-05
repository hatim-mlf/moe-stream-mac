def max_bracket_depth(text):
 s=0; m=0
 for c in text:
  if c in "([{": s+=1; m=max(m,s)
  elif c in ")]}" : s-=1
 return m
