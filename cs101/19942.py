m,n,p,q=map(int,input().split())
A=[[int(i) for i in map(int,input().split())] for _ in range(m)]
B=[[int(i) for i in map(int,input().split())] for _ in range(p)]
C=[[0 for i in range(n+1-q)] for j in range(m+1-p) ]

for i in range(m+1-p):
    for j in range(n+1-q):
        for a in range(i,i+p):
            for b in range(j,j+q):
                C[i][j]+=A[a][b]*B[a-i][b-j]
for c in C :
    print(*c)
                
