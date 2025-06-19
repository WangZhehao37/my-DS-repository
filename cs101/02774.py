from typing import List


class Solution:
    def binary_search_iterative(self, woods: List[int], target: int) -> int:
        left, right = 1, max(woods)
        result = 0
        while left <= right:
            countcut = 0
            mid = left + (right - left) // 2
            for i in woods:
                countcut += i//mid
            if countcut >= target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result


if __name__ == "__main__":
    N, K = map(int, input().split())
    woods = []
    for i in range(N):
        woods.append(int(input()))
    ans = Solution().binary_search_iterative(woods, K)
    print(ans)
