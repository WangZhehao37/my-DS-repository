from math import sqrt
n=int(input())
ans=[]
for i in range(n):
    a,b,c=map(float,input().split())
    if b == 0 :
        b=-b
    if b**2-4*a*c > 0 :
        x1=(-b + sqrt(b*b-4*a*c))/(2*a)
        x2=(-b - sqrt(b*b-4*a*c))/(2*a)
        if  x1<x2:
            x1,x2=x2,x1
        ans.append(f"x1={'{:.5f}'.format(x1)};x2={'{:.5f}'.format(x2)}")
    elif b**2-4*a*c == 0 :
        ans.append(f"x1=x2={'{:.5f}'.format(-b/(2.00000*a))}")
    elif b**2-4*a*c < 0 :
        ans.append(f"x1={'{:.5f}'.format(-b/(2*a))}+{'{:.5f}'.format(sqrt(4*a*c-b*b)/(2*a))}i;x2={'{:.5f}'.format(-b/(2*a))}-{'{:.5f}'.format(sqrt(4*a*c-b*b)/(2*a))}i")
for i in ans:
    print(i)
    