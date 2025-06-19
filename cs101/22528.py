import math
scores = list(map(float, input().split()))
scores.sort(reverse=-1)
n = len(scores)
A = math.ceil(0.6*n)
s_A = scores[A-1]


def find_min_satisfy_condition(x,target):
    left, right = 0, 1000000000
    ans = -1
    while left <= right:
        mid = (left + right) // 2
        a=mid/1000000000
        if a*x+1.1**(a*x) >=target:
            ans = mid
            right = mid - 1  # 继续寻找更小的解
        else:
            left = mid + 1
    return ans

print(find_min_satisfy_condition(s_A,85))