from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val: str):
        self.val = val
        self.child = []

    def add_child(self, child: 'TreeNode'):
        self.child.append(child)


class Solution:
    def build_tree(self, s: list) -> Optional[TreeNode]:
        if not s:
            return None

        root = TreeNode(s[0])
        queue = deque([(root, int(s[1]))])
        i = 1

        while i < len(s)//2:
            node, degree = queue.popleft()
            for _ in range(degree):
                child = TreeNode(s[2*i])
                node.add_child(child)
                queue.append((child, int(s[2*i+1])))
                i += 1

        return root

    def trans_order(self, node: Optional[TreeNode]) -> List[str]:
        if node is None:
            return []

        result = []
        for child in node.child:
            result.extend(self.trans_order(child))

        result.append(node.val)
        return result


if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        s = input().split()
        root = Solution().build_tree(s)
        result = Solution().trans_order(root)
        print(' '.join(result), end=' ')
