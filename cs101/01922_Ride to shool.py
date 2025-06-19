from math import ceil
while True:
    n=int(input())
    t=[]
    if n == 0 :
        break
    for i in range(n):
        a,b=map(int,input().split())
        if b>=0:
            t.append(ceil(b+4.5/a*3600))
    print(min(t))