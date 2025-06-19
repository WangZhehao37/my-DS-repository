from collections import deque
from typing import List


class Solution:
    def YSF(self, n: int, k: int) -> List[int]:
        res = []
        queue = deque([i for i in range(1, n+1)])
        while len(queue) > 1:
            queue.rotate(-(k-1))
            a = queue.popleft()
            res.append(a)
        return res


if __name__ == "__main__":
    N, K = map(int, input().split())
    ans = Solution().YSF(N, K)
    print(*ans)
