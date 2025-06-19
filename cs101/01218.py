n=int(input())
for i in range(n):
    a=int(input())
    pr=[1]*a
    for j in range(2,a+1):
        for k in range(j-1,a,j):
            if pr[k]==1:
                pr[k]=0
            else :
                pr[k]=1
    print(sum(pr))
                