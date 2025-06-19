n,m=map(int,input().split())
A=[[0 for j in range(m)] for i in range(n+1)]
B=[[0 for j in range(m+1)] for i in range(n)]
for i in range(n):
    inp=list(map(int,input().split()))
    for j in range(m):
        if inp[j]==1:
            A[i][j]+=1
            A[i+1][j]+=1
            B[i][j]+=1
            B[i][j+1]+=1
count=0
for a in A:
    count+=a.count(1)
for b in B:
    count+=b.count(1)
print(count)
    