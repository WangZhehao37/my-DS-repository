import heapq
v, a = map(int, input().split())
adj = [[] for i in range(v+1)]
in_degree = [0]*(v+1)
for i in range(a):
    u, w = map(int, input().split())
    adj[u].append(w)
    in_degree[w] += 1

heap = []
for i in range(1, v+1):
    if in_degree[i]==0:
        heapq.heappush(heap,i)
        
result = []
while heap:
    u=heapq.heappop(heap)
    result.append(f'v{u}')
    for neib in adj[u]:
        in_degree[neib]-=1
        if in_degree[neib]==0:
            heapq.heappush(heap,neib)
            
print(' '.join(result))