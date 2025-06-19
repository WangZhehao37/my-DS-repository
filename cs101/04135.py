N, M = map(int, input().split())
cost = []
for i in range(N):
    cost.append(int(input()))
left, right, ans = max(cost), sum(cost), 0
while left <= right:
    mid = (right+left)//2
    i = 0
    count_ = 0
    count_fajo = 1
    while i < N:
        count_ += cost[i]
        if count_ > mid:
            count_ = 0
            count_fajo += 1
        else:
            i += 1
    if count_fajo <= M:
        right = mid-1
        ans = mid
    else:
        left = mid+1
print(ans)
