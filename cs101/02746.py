def Y(a,b):
    if a==1:
        return 1
    else :
        return (Y(a-1,b)-1+b) % a +1

while True:
    n,m=map(int,input().split())
    if n == 0 and m == 0 :
        break
    else :
        print(Y(n,m))

        