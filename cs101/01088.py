R,C=map(int,input().split())
move=[(1,0),(-1,0),(0,1),(0,-1)]
Map=[]
for i in range(R):
    Map.append([int(j) for j in input().split()])
dp=[[-1]*C for _ in range(R)]
max_=0
def foundmax(Map,x,y):
    R=len(Map)
    C=len(Map[0])
    if dp[x][y]!=-1:
        return dp[x][y]
    lenmax=1
    for dx,dy in move:
        x1,y1=x+dx,y+dy
        if 0<=x1<R and 0<=y1<C and Map[x1][y1]<Map[x][y]:
            lenmax=max(lenmax,foundmax(Map,x1,y1)+1)
    dp[x][y]=lenmax
    return lenmax
for i in range(R):
    for j in range(C):
        max_=max(max_,foundmax(Map,i,j))
print(max_)