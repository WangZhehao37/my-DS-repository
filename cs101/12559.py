import math
n = int(input())
A = input().split()
m = max(len(A[i]) for i in range(n))
A.sort(key=lambda x: x*(math.ceil(2*m/len(x))),reverse=True)
print("".join(A), "".join(reversed(A)))
