import heapq
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
m, n, p = map(int, input().split())
map_ = []


def dis(map_, x1, y1, x2, y2):
    m, n = len(map_), len(map_[0])
    distance = [[float('inf')]*n for _ in range(m)]
    distance[x1][y1] = 0
    visited = set()
    pre = [(0, x1, y1)]
    while pre:
        distan, x, y = heapq.heappop(pre)
        if x == x2 and y == y2:
            return distan
        if (x,y) in visited:
            continue
        visited.add((x,y))
        for dx,dy in moves:
            x_,y_=x+dx,y+dy
            if 0<=x_<m and 0<=y_<n and map_[x_][y_]!="#":
                add=abs(int(map_[x][y])-int(map_[x_][y_]))
                if add+distan<distance[x_][y_]:
                    distance[x_][y_]=add+distan
                    heapq.heappush(pre,(distance[x_][y_],x_,y_))
    return "NO"

for i in range(m):
    map_.append(input().split())
for _ in range(p):
    x_0, y_0, X, Y = map(int, input().split())
    if map_[x_0][y_0]=="#" or map_[X][Y]=="#" :
        print("NO")
    else :
        print(dis(map_,x_0,y_0,X,Y))
