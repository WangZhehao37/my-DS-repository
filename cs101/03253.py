while True:
    n,p,m=map(int,input().split())
    if n == 0 and m == 0 and p==0:
        break
    else :
        list=[i for i in range(1,n+1)]
        for _ in range(1,n):
            a=list.pop((p+m-2)%(n-_+1))
            p=(p+m-2)%(n-_+1)+1
            print(f"{a},",end="")
        print(list[0])
            