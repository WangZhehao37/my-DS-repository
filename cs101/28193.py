n,m=map(int,input().split())
adj=[[] for _ in range(n+1)]
visited=[False]*(n+1)
visited[0]=True
count_tot=0

ci=[0]
for i in input().split():
    ci.append(int(i))
    
for i in range(m):
    x,y=map(int,input().split())
    adj[x].append(y)
    adj[y].append(x)
    
def cal(mincost,x):
    mincost=min(mincost,ci[x])
    for y in adj[x]:
        if visited[y] == False:
            visited[y]=True
            mincost=min(mincost,cal(mincost,y))
    return mincost

for j in range(n+1):
    if visited[j]==False:
        count_tot+=cal(float("inf"),j)
        
print(count_tot)