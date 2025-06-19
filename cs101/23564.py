from math import sqrt
n=int(input())
sqrt_n=int(sqrt(n))
factor=[]
for check in range(2,sqrt_n+1):
    while n % check == 0:
        factor.append(check)
        n/=check
if n > 1:
    factor.append(n)
factor=sorted(factor)
factor_setted=set(factor)
if len(factor_setted)!=len(factor):
    print(0)
elif len(factor) % 2 == 0:
    print(1)
else: print(-1)