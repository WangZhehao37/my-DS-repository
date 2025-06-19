n,w=map(int,input().split())
val=0
A=[[0]*3 for i in range(n)]
for i in range(n):
    A[i][0],A[i][1]=map(int,input().split())
    A[i][2]=A[i][0]/A[i][1]
A=sorted(A,key=lambda x: x[2],reverse=True)
for i in range(n):
    if w >= A[i][1] :
        w=w-A[i][1]
        val+=A[i][0]
    else :
        val+=A[i][0]*w/A[i][1]
        break
print(f"{val:.1f}")