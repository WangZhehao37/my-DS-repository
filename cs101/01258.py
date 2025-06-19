import heapq


def prim(map, n):
    heap = []
    visited = [False] * n
    tot = 0
    mst = []

    visited[0] = True
    for i in range(n):
        heapq.heappush(heap, (map[0][i], 0, i))

    while heap and len(mst) < n-1:
        weight, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v]= True
        mst.append((weight, u, v))
        tot+=weight
        for i in range(n):
            heapq.heappush(heap, (map[v][i], v, i))
            
    return tot


while True:
    try:
        N = int(input())
        matrix = []
        for i in range(N):
            matrix.append(list(map(int, input().split())))
        print(prim(matrix,N))
    except EOFError:
        break