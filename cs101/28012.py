from collections import deque

n=int(input())
adj=[[] for _ in range(n)]
visited=[False]*n
for i in range(n-1):
    a,b=map(int,input().split())
    adj[a].append(b)
    adj[b].append(a)
cant=[int(j) for j in input().split()]

count=0
queue=deque()

queue.append(0)
visited[0]=True
count+=1

while queue:
    dot=queue.popleft()
    for i in adj[dot]:
        if visited[i]==False and i not in cant :
            count+=1
            queue.append(i)
            visited[i]=True
            
print(count)