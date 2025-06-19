a1,a2=map(int,input().split())
A=[list(map(int,input().split())) for i in range(a1)]
b1,b2=map(int,input().split())
B=[list(map(int,input().split())) for i in range(b1)]
c1,c2=map(int,input().split())
C=[list(map(int,input().split())) for i in range(c1)]
D=[[0]*c2 for i in range(c1)]
if a1==c1 and a2==b1 and b2==c2:
    #第i行第j列
    for i in range(c1):
        for j in range(c2):
            for k in range(a2):
                D[i][j]+=A[i][k]*B[k][j]
            D[i][j]+=C[i][j]
    for i in D:
        print(*i)
else:
    print("Error!")
        
