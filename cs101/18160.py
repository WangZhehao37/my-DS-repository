def dps(mat,visited,N,M,n,m):
    if n>=N or n<0 or m>=M or m<0 or visited[n][m]==True or mat[n][m]==".":
        return 0
    visited[n][m]=True
    area=1
    arrows=[(1,0),(1,-1),(1,1),(-1,0),(-1,-1),(-1,1),(0,1),(0,-1)]
    for dx,dy in arrows:
        area+=dps(mat,visited,N,M,n+dx,m+dy)   
    return area
        
def findmax(mat,visited,N,M):
    max_=0
    for i in range(N):
        for j in range(M):
            if mat[i][j]=="W" and visited[i][j]==False:
                max_=max(max_,dps(mat,visited,N,M,i,j))
    return max_


T=int(input())
for i in range(T):
    N,M=map(int,input().split())
    mat=[]
    for i in range(N):
        mat.append(input())
    visited=[[False for j in range(M)] for i in range(N)]
    print(findmax(mat,visited,N,M))
        