import heapq

def dijkstra_with_cost(graph, start, K):
    n = len(graph)
    INF = float('inf')
    dist = [[INF] * (K + 1) for _ in range(n)]
    dist[start][0] = 0  
    heap = [(0, start, 0)] 

    while heap:
        d, u, c = heapq.heappop(heap)
        if d > dist[u][c]:
            continue  
        for v, l, t in graph[u]:
            new_c = c + t
            if new_c > K:
                continue  
            new_d = d + l
            if new_d < dist[v][new_c]:
                dist[v][new_c] = new_d
                heapq.heappush(heap, (new_d, v, new_c))

    min_len = min(dist[n - 1][c] for c in range(K + 1) if dist[n - 1][c] < INF)
    return min_len if min_len < INF else -1

K = int(input())
N = int(input())
R = int(input())
graph = [[] for _ in range(N)]

for _ in range(R):
    S, D, L, T = map(int, input().split())
    graph[S - 1].append((D - 1, L, T))  


print(dijkstra_with_cost(graph, 0, K))