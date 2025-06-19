import math
#这里默认了x大于等于3
def judge(x):
    for i in range(2,math.isqrt(x)+1):
        if x % i == 0 :
            return 0
    return 1
n=int(input())
if n<6 or n % 2 ==1 :
    print("Error!")
else:
    for i in range(3,n//2+1):
        if judge(i) and judge(n-i) :
            print(f"{n}={i}+{n-i}")