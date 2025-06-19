def judge(a, i, n):
    for j in range(n):
        if int(a[j]) == i or int(a[j])-j==i-n or int(a[j])+j==i+n:
            return False 
    return True

def track(a, n):
    if n == 8:
        return [a]
    solutions = []
    for i in range(1, 9):
        if judge(a, i, n):
            solutions += track(a+str(i),n+1)
    return solutions

b=int(input())

result=track('',0)

for i in range(b):
    m=int(input())
    print(result[m-1])