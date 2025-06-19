n=int(input())
A=[[0 for _ in range(n)] for __ in range(n)]
i=1
up=0
left=0
right=n-1
down=n-1
while True:
    if up==down+1:
        break
    for j in range(left,right+1):
        A[up][j]=i
        i+=1
    up+=1
        
    if left==right+1:
        break
    for j in range(up,down+1):
        A[j][right]=i
        i+=1
    right-=1
    
    if up==down+1:
        break
    for j in range(right,left-1,-1):
        A[down][j]=i
        i+=1
    down-=1
    
    if left==right+1:
        break
    for j in range(down,up-1,-1):
        A[j][left]=i
        i+=1
    left+=1
    
for a in A:
    print(*a)
    