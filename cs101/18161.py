row_A,col_A=map(int,input().split())
A=[[int(i) for i in input().split()] for _ in range(row_A)]
row_B,col_B=map(int,input().split())
B=[[int(i) for i in input().split()] for _ in range(row_B)]
row_C,col_C=map(int,input().split())
C=[[int(i) for i in input().split()] for _ in range(row_C)]
if col_A==row_B and row_A==row_C and col_B==col_C :
    D=[[0 for i in range(col_C)] for _ in range(row_C)]
    for i in range(row_C):
        for j in range(col_C):
            for k in range(col_A):
                D[i][j]+=A[i][k]*B[k][j]
            D[i][j]+=C[i][j]
    for i in range(row_C):
        print(*D[i])
else :
    print("Error!")