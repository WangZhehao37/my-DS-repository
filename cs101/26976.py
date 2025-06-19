n = int(input())
A = [int(i) for i in input().split()]
B = [(1,1) for i in range(n)]
max_=1
for i in range(1,n):
    for j in range(i,-1,-1):
        if A[i]-A[j]<0 and B[j][1]+1>B[i][0]:
            B[i]=(B[j][1]+1,B[i][1])
            max_=max(max_,B[j][1]+1)
        elif A[i]-A[j]>0 and B[j][0]+1>B[i][1]:
            B[i]=(B[i][0],B[j][0]+1)
            max_=max(max_,B[j][0]+1)
print(max_)
