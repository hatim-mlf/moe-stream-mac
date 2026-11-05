import heapq

def stable_topological_order(nodes, edges):
    order = list(nodes)
    try:
        position = {node: i for i, node in enumerate(order)}
    except TypeError as exc:
        raise ValueError('nodes must be hashable') from exc
    if len(position) != len(order):
        raise ValueError('duplicate node')
    following = [[] for _ in order]
    indegree = [0] * len(order)
    for edge in edges:
        try:
            before, after = edge
            i, j = position[before], position[after]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError('invalid edge') from exc
        following[i].append(j)
        indegree[j] += 1
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        i = heapq.heappop(ready)
        result.append(order[i])
        for j in following[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                heapq.heappush(ready, j)
    if len(result) != len(order):
        raise ValueError('cycle')
    return result
