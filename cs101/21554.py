n=int(input())
A=input().split()
for i in range(n):
    A[i]=(int(A[i]),i+1)
A.sort()
sum=0
for i in range(n):
    sum+=A[i][0]*(n-i-1)
for i in range(n):
    print(A[i][1],end=" ")
print("")
print(f"{sum/n:.2f}")