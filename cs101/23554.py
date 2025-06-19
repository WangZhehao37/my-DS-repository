n=int(input())
a=[i for i in range(1,n+1)]
b=[]
c=[int(i) for i in input().split()]
for i in c :
    if i in a:
        a.remove(i)
    else :
        b.append(i)
print(*a)
print(*sorted(b))
