def max_bracket_depth(text):
 stack=[]; best=0; pair={")":"(","]":"[","}":"{"}
 for c in text:
  if c in "([{": stack.append(c); best=max(best,len(stack))
  elif c in ")]}":
   if stack and stack.pop()!=pair[c]: return -1
 return best
