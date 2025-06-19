from math import gcd #gcd 是取最大公约数
while(True):
    try:
        a,b=map(int,input().split())
        print(gcd(a,b))
    except EOFError:
        break
    
#自己写一个
"""def gcd_(a,b):
    n=0
    for i in range(1,min(a,b)+1):
        if a%i==0 and b%i==0:
            n=i
    return n"""