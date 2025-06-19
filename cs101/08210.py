L, N, M = map(int,input().split())
site=[0]
for i in range(N):
    site.append(int(input()))
site.append(L)
l=[]
for i in range(N+1):
    l.append(site[i+1]-site[i])

def movecount(min_,N):
    current=0
    count_=0
    for i in range(N+1):
        current+=l[i]
        if current<min_:
            count_+=1
        else:
            current=0
    return count_
up=L
down=1

while True:
    if up==down+1:
        break
    a=movecount((up+down)//2,N)
    if a<=M:
        down=(up+down)//2
    else :
        up=(up+down)//2
        
print(down)