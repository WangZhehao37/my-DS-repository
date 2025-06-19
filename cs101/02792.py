n=int(input())
for i in range(n):
    s=int(input())
    
    a=int(input())
    A=sorted(list(map(int,input().split())))
    A_1=[(A[0],1)]
    for i in range(1,a):
        if A[i]==A[i-1]:
            A_1[-1]=(A_1[-1][0],A_1[-1][1]+1)
        else :
            A_1.append((A[i],1))
            
    b=int(input())
    B=sorted(list(map(int,input().split())),reverse=True)
    B_1=[(B[0],1)]
    for i in range(1,b):
        if B[i]==B[i-1]:
            B_1[-1]=(B_1[-1][0],B_1[-1][1]+1)
        else :
            B_1.append((B[i],1))
            
    a1=0
    b1=0
    sum_=0
    while a1<len(A_1) and b1<len(B_1):
        if A_1[a1][0]+B_1[b1][0]==s:
            sum_+=A_1[a1][1]*B_1[b1][1]
            a1+=1
            b1+=1
        elif A_1[a1][0]+B_1[b1][0]<s:
            a1+=1
        elif A_1[a1][0]+B_1[b1][0]>s:
            b1+=1
    print(sum_)
        