def Pell(n):
    dp=[0]*(n+1)
    dp[1]=1
    dp[2]=2
    
    for i in range(3,n+1):
        dp[i]=(2*dp[i-1]+dp[i-2])%32767
    return dp
n=int(input())
k=[]
for i in range(n):
    k.append(int(input()))
s=Pell(max(k))
for i in k:
    print(s[i])