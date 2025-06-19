def Vb(x):
    if x==1 or x==2 :
        return 1
    else :
        return Vb(x-1)+Vb(x-2)
n=int(input())
for i in range(n):
    print(Vb(int(input())))