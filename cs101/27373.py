m=int(input())
n=int(input())
num=input().split()
for i in range(n-1):
    for j in range(n-1-i):
        if num[j]+num[j+1] > num[j+1]+num[j] :
            num[j],num[j+1]=num[j+1],num[j]
lens=[0]*n
for i in range(n):
    lens[i]=len(num[i])
#建立dp数组：前i个，位数不超过j的数字
dp=[['']*(m+1) for i in range(n+1)]
for i in range(1, n + 1):
        for w in range(m + 1):
            if lens[i - 1] <= w:
                # 如果可以放入第i个物品，则选择放或不放该物品中的较大值
                dp[i][w] = str(max(int('0'+dp[i - 1][w]), int(num[i - 1]+dp[i - 1][w - lens[i - 1]] )))
            else:
                # 如果不能放入第i个物品，则继承之前的结果
                dp[i][w] = dp[i - 1][w]
print(dp[n][m])