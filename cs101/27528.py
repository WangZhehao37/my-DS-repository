n=int(input())
steps=[0]*(n+1)
steps[1]=1
for i in range(2,n+1):
    steps[i]=2*steps[i-1]
print(steps[n])