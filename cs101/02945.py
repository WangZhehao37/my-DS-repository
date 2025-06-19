k=int(input())
dam=list(map(int,input().split()))
dp=[1 for i in range(k)]
for i in range(k):
    for j in range(i):
        if dam[j]>=dam[i]:
            dp[i]=max(dp[i],dp[j]+1)
print(max(dp))
        