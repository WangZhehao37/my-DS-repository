from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root



def cengorder(root):
    if not root:
        return []
    
    queue=deque() 
    queue.append(root)
    ans=[]
    
    while queue:
        node=queue.popleft()
        ans.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
        
    return ans


listmap=[int(i) for i in input().split()]
root=None
for i in listmap:
    root=insert(root,i)
print(*cengorder(root))
    
