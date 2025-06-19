n,m=map(int,input().split())
A=sorted([int(i) for i in input().split()])
B=sorted([A[i]-A[i-1] for i in range(1,n)])
print(A[-1]-A[0]-sum(B[-1:-m:-1]))

