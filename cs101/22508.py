from collections import deque

def topological_sort_kahn(graph,n):
    in_degree = {u: 0 for u in range(n)}
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = []
    longest=[0]*n

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if longest[v] < longest[u] + 1:
                longest[v] = longest[u] + 1
                
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != len(graph):
        return None  # 存在环路
    return sum(longest)+100*n


n,m=map(int,input().split())
graph=[[] for i in range(n)]
for i in range(m):
    u,v=map(int,input().split())
    if u not in graph[v]:
        graph[v].append(u)


print(topological_sort_kahn(graph,n)) 