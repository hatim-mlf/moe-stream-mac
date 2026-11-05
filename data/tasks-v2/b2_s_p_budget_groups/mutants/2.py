def split_by_budget(costs,limit):
 if not isinstance(limit,int) or isinstance(limit,bool) or limit<=0: raise ValueError()
 groups=[]; cur=[]; total=0
 for cost in costs:
  if not isinstance(cost,int) or isinstance(cost,bool) or cost<0 or cost>limit: raise ValueError()
  if cur and total+cost>limit: groups.append(cur);cur=[];total=0
  cur.append(cost);total+=cost
 return groups
