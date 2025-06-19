import sys


def main():
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    vals = list(map(int, data[1:]))

    dp = [[0, 0] for i in range(n+2)]

    for i in range(n, 0, -1):
        val = vals[i-1]
        left = 2*i
        right = 2*i+1

        select = val
        if left <= n:
            select += dp[left][0]
        if right <= n:
            select += dp[right][0]
        dp[i][1] = select

        not_select=0
        max_left = max(dp[left][0], dp[left][1]) if left <= n else 0
        max_right = max(dp[right][0], dp[right][1]) if right <= n else 0
        dp[i][0] = max_left + max_right
        
        print(max(dp[1][0],dp[1][1]))
        
if __name__ =="__main__":
    main()