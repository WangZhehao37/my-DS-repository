from collections import deque
n,m=map(int,input().split())
ai=input().split()
queue=deque()
for i in range(n):
    queue.append((i+1,int(ai[i])))

while queue:
    num,need=queue.popleft()
    need_=need-m
    if need_<=0:
        lastout=num
    else:
        queue.append((num,need_))
        
print(lastout)