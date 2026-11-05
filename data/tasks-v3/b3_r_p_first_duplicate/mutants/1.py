def first_duplicate_item(values):
 seen=set(); last=None
 for v in values:
  if v in seen:last=v
  seen.add(v)
 return last
