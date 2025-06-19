n=int(input())
for i in range(n):
    l,num=map(int,input().split())
    ants=list(map(int,input().split()))
    min_=0
    max_=0
    for i in ants:
        if min(i,l-i) > min_ :
            min_ =min(i,l-i)
        if max(i,l-i) > max_ :
            max_=max(i,l-i)
    print(min_,max_)