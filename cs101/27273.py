from math import log
t=int(input())
ans=[]
for i in range(t):
    n=int(input())
    m=int(log(n,2))
    ans.append(1/2*n*(n+1)+2-2**(m+2))
for i in ans:
    print(int(i))