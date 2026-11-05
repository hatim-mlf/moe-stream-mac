import heapq

def stable_topological_order(nodes, edges):
    order=list(nodes); pos={v:i for i,v in enumerate(order)}
    nxt=[[] for _ in order]; deg=[0]*len(order)
    for a,b in edges:
        i,j=pos[a],pos[b]; nxt[i].append(j); deg[j]+=1
    q=[i for i,d in enumerate(deg) if d==0]; heapq.heapify(q); out=[]
    while q:
        i=heapq.heappop(q); out.append(order[i])
        for j in nxt[i]:
            deg[j]-=1
            if deg[j]==0: heapq.heappush(q,j)
    if len(out)!=len(order): raise ValueError('cycle')
    return out
