import heapq
from collections import defaultdict

def dijkstra(graph, start, end):
    # 初始化距离和访问标记
    dist = {node: float('inf') for node in range(1,end+1)}
    dist[start] = 0
    visited = set()
    priority_queue = [(0, start)]  # (距离, 节点)

    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph[u]:
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                heapq.heappush(priority_queue, (dist[v], v))
    
    return dist[end]

N,M=map(int,input().split())
graph=defaultdict(list)
for i in range(M):
    A,B,c=map(int,input().split())
    graph[A].append((B,c))
print(dijkstra(graph,1,N))