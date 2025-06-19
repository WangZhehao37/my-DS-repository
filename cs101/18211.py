p=int(input())
price=[int(i) for i in input().split()]
price.sort()
i=0
j=len(price)-1
W=0
D=0
while i<=j:
    if p>=price[i]:
        W+=1
        p-=price[i]
        i+=1
    else :
        if W==D:
            break
        elif j==i:
            break
        else :
            p+=price[j]
            j-=1
            D+=1
print(W-D)
    