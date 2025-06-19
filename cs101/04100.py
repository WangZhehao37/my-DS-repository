k=int(input())
for _ in range(k):
    n=int(input())
    ans=0
    A=[tuple(map(int,input().split())) for i in range(n)]
    A.sort()
    t=0
    t_=100
    for i in A:
        if i[0]==t:
            continue
        if i[0]>t:
            t=i[0]
            if i[1]<t_:
                t_=i[1]
        if i[0]>t_:
            t=i[0]
            ans+=1
            t_=i[1]
    print(ans+1)