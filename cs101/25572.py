n=int(input())
move=[(1,0),(-1,0),(0,1),(0,-1)]
Map=[]
ori=[]
for i in range(n):
    Map.append([int(j) for j in input().split()])
for i in range(n):
    for j in range(n):
        if Map[i][j]==9:
            X=i
            Y=j
        if Map[i][j]==5:
            ori.append((i,j))
visited=[[False]*n for i in range(n)]

#横着的
def findpath_heng(x,y,n):
    visited[x][y]=True
    cou=0
    if Map[x][y]==9 or Map[x][y+1]==9:
        return 1
    for dx,dy in move:
        x1,y1=x+dx,y+dy
        if 0<=x1<n and 0<=y1<n-1 and Map[x1][y1]!=1 and Map[x1][y1+1]!=1 and visited[x1][y1]==False:
            cou+=findpath_heng(x1,y1,n)
    return cou

def findpath_shu(x,y,n):
    visited[x][y]=True
    cou=0
    if Map[x][y]==9 or Map[x+1][y]==9:
        return 1
    for dx,dy in move:
        x1,y1=x+dx,y+dy
        if 0<=x1<n-1 and 0<=y1<n and Map[x1][y1]!=1 and Map[x1+1][y1]!=1 and visited[x1][y1]==False:
            cou+=findpath_shu(x1,y1,n)
    return cou

if ori[0][0]==ori[1][0]:
    if findpath_heng(ori[0][0],min(ori[0][1],ori[1][1]),n)>0:
        print('yes')
    else:
        print('no')
if ori[0][1]==ori[1][1]:
    if findpath_shu(min(ori[0][0],ori[1][0]),ori[0][1],n)>0:
        print('yes')
    else:
        print('no')