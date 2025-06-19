while True :
    n=int(input())
    sum_=0
    if n == 0 :
        break
    else :
        A=[]
        for i in range(n):
            a,b=map(int,input().split())
            A.append((a,-b))
        A.sort()
        A.append((-1,-1))
        cost=-float("inf")
        for i in range(n):
            if A[i][0]==A[i+1][0]:
                continue
            if A[i][1] > cost :
                cost=A[i][1]
                sum_+=1
        print(sum_)
            