from collections import deque
def reach(m,n,map):
    queue=deque([(0,0,0)])
    visited=[[False]*n for _ in range(m)]
    visited[0][0]=True
    move=[(1,0),(-1,0),(0,-1),(0,1)]
    while queue:
        x,y,step=queue.popleft()
        #x:行 y:列
        if map[x][y]==1:
            return step
        for dx,dy in move :
            x1=x+dx
            y1=y+dy
            if 0<=x1<m and 0<=y1<n and (not visited[x1][y1]==True) and (not map[x1][y1]==2):
                visited[x1][y1]=True
                queue.append((x1,y1,step+1))
    return "NO"

m,n=map(int,input().split())
map=[]
for i in range(m):
    map.append([int(i) for i in input().split()])
print(reach(m,n,map))
        