n,b=map(int,input().split())
price=list(map(int,input().split()))
weigh=list(map(int,input().split()))
dp=[[0 for i in range(b+1)] for j in range(n+1) ]
for i in range(1,n+1):
    for j in range(b+1):
        dp[i][j]=dp[i-1][j]
        if j>=weigh[i-1]:
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-weigh[i-1]]+price[i-1])
print(dp[n][b])