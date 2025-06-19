m=int(input())
A=[[int(i) for i in input().split()] for j in range(m)]
for i in range(m):
    A[i]=sorted(A[i])
    if sum(A[i])<24:
        print("NO")
    elif 2*A[i][3]-sum(A[i])>24:
        print("NO")
    
    elif sum(A[i])==24:
        print("YES")
    
    elif sum(A[i])-2*A[i][0]==24:
        print("YES")
    elif sum(A[i])-2*A[i][1]==24:
        print("YES")
    elif sum(A[i])-2*A[i][2]==24:
        print("YES")
    elif sum(A[i])-2*A[i][3]==24:
        print("YES")
        
    elif sum(A[i])-2*A[i][0]-2*A[i][1]==24:
        print("YES")
    elif sum(A[i])-2*A[i][0]-2*A[i][2]==24:
        print("YES")
    elif sum(A[i])-2*A[i][1]-2*A[i][2]==24:
        print("YES")
    elif sum(A[i])-2*A[i][3]-2*A[i][0]==24:
        print("YES")
        
    elif 2*A[i][3]-sum(A[i])==24:
        print("YES")
        
    else:
        print("NO")