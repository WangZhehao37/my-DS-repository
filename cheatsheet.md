## 欧拉筛法
#### **代码实现（Python）**
```python
def euler_sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0:2] = [False, False]  # 0 和 1 不是素数
    primes = []  # 存储质数
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)  # 将素数加入列表
        # 用当前已找到的素数筛除合数
        for p in primes:
            if i * p > n:
                break  # 超出范围时提前退出
            is_prime[i * p] = False
            if i % p == 0:
                break  # 确保每个合数只被最小质因数筛除
    return primes
```

#### **示例**
```python
print(euler_sieve(30))  
# 输出: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```


## 排序

### **1. 冒泡排序（Bubble Sort）**
- **基本思想**：重复遍历待排序序列，比较相邻元素，若逆序则交换，直到无交换发生。
- **时间复杂度**：
  - 最坏/平均：$ O(n^2) $
  - 最佳：$ O(n) $（已有序时）
#### 代码示例
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # 提前终止优化
    return arr
```

---

### **2. 快速排序（Quick Sort）**
- **基本思想**：分治法。选取基准元素，将数组分为两部分，左边小于基准，右边大于基准，递归排序子数组。
- **时间复杂度**：
  - 平均：$ O(n \log n) $
  - 最坏：$ O(n^2) $（完全有序时）
#### 代码示例
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```
---

### **3. 插入排序（Insertion Sort）**
- **基本思想**：将未排序元素逐个插入到已排序序列的适当位置。
- **时间复杂度**：
  - 最坏/平均：$ O(n^2) $
  - 最佳：$ O(n) $（已有序时）
#### 代码示例
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```
---

### **4. 归并排序（Merge Sort）**
- **基本思想**：分治法。将数组分成两半，分别排序后合并两个有序子数组。
- **时间复杂度**：始终 $ O(n \log n) $
#### 代码示例
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
```
---

## **双指针**

双指针算法（Two Pointers Algorithm）核心思想是通过**两个指针的协同移动**来减少时间复杂度，通常将暴力解法的 $ O(n^2) $ 优化到 $ O(n) $。

---

#### **一、双指针的分类**
根据指针的移动方向和策略，双指针主要分为两类：

1. **快慢指针（Fast & Slow Pointers）**
   - **特点**：两个指针从同一位置出发，快指针每次移动步长较大，慢指针较小。
   - **应用场景**：
     - 检测链表是否有环（Floyd 判圈算法）。
     - 查找链表中点。
     - 删除重复元素。
     - 数组去重。

2. **对撞指针（Two Sum Pointers）**
   - **特点**：两个指针分别从数组两端向中间移动。
   - **应用场景**：
     - 有序数组中两数之和。
     - 三数之和、四数之和。
     - 反转字符串。
     - 最大容器面积问题。

---

#### **二、典型应用场景与代码示例**

##### **1. 快慢指针：链表是否有环**
**问题**：判断链表中是否存在环。  
**思路**：快指针每次走两步，慢指针每次走一步。若有环，快指针最终会追上慢指针。

```python
def has_cycle(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True  # 有环
    return False
```

---

##### **2. 对撞指针：有序数组中两数之和**
**问题**：在有序数组中找到两个数，使它们的和等于目标值。  
**思路**：左指针从左向右，右指针从右向左移动，根据和的大小调整指针。

```python
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]
```

---

##### **3. 快慢指针：删除数组重复元素**
**问题**：删除排序数组中的重复项，保持相对顺序。  
**思路**：慢指针记录非重复元素的位置，快指针遍历数组。

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

---


##### **4. 滑动窗口（双指针变体）：最长无重复子串**
**问题**：找出字符串中无重复字符的最长子串长度。  
**思路**：右指针扩展窗口，左指针收缩窗口以去除重复字符。

```python
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

---

## 二分查找
---
### 通用二分查找模板（闭区间 [left, right]）

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # 未找到
```

####  说明
- 使用闭区间 `[left, right]`，即 `left <= right`。
- 每次计算中间点 `mid = left + (right - left) // 2`，避免溢出。
- 如果 `nums[mid] == target`，直接返回 `mid`。
- 如果 `nums[mid] < target`，说明目标在右半部分，设置 `left = mid + 1`。
- 如果 `nums[mid] > target`，说明目标在左半部分，设置 `right = mid - 1`。
- 循环结束未找到目标，返回 `-1`。

---

### 变体模板一：查找第一个等于 target 的元素索引

```python
def find_first(nums, target):
    left, right = 0, len(nums) - 1
    res = -1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            res = mid
            right = mid - 1  # 继续向左找是否有更小的索引
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return res
```

####  说明
- 与通用模板不同的是，找到 `nums[mid] == target` 后不立即返回，而是继续向左搜索。
- 最终 `res` 会保存第一个等于 `target` 的索引。

---

### 变体模板二：查找最后一个等于 target 的元素索引

```python
def find_last(nums, target):
    left, right = 0, len(nums) - 1
    res = -1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            res = mid
            left = mid + 1  # 继续向右找是否有更大的索引
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return res
```

####  说明
- 找到 `nums[mid] == target` 后继续向右搜索。
- 最终 `res` 会保存最后一个等于 `target` 的索引。

---

###  变体模板三：查找第一个大于等于 target 的元素索引（插入位置）

```python
def find_first_ge(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left  # left 是第一个大于等于 target 的位置
```

####  说明
- 不关心 `nums[mid]` 是否等于 `target`，只关心 `nums[mid] < target`。
- 循环结束后 `left` 是第一个大于等于 `target` 的元素索引。
- 也可以用于查找 `target` 的插入位置。

---

###  变体模板四：查找最后一个小于等于 target 的元素索引

```python
def find_last_le(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return right  # right 是最后一个小于等于 target 的位置
```

#### 说明
- 不关心 `nums[mid]` 是否等于 `target`，只关心 `nums[mid] > target`。
- 循环结束后 `right` 是最后一个小于等于 `target` 的元素索引。

---

### 示例测试

```python
arr = [1, 2, 3, 4, 4, 5, 6]

print(binary_search(arr, 4))       # 输出 3（任意一个 4）
print(find_first(arr, 4))          # 输出 3（第一个 4）
print(find_last(arr, 4))           # 输出 4（最后一个 4）
print(find_first_ge(arr, 4))       # 输出 3（第一个大于等于 4）
print(find_last_le(arr, 4))        # 输出 4（最后一个小于等于 4）
```

---

###  注意事项

- **数组必须是有序的**，否则二分查找无法正确运行。
- **边界条件处理** 是关键，尤其是 `mid` 的计算方式和 `left`、`right` 的更新策略。
- **避免死循环**，确保每次循环都缩小了查找范围。
- **mid 的计算方式**：推荐使用 `mid = left + (right - left) // 2`，防止溢出。

---
这段代码实现了著名的**骑士巡游问题（Knight's Tour）**，即在 n×n 的棋盘上，骑士从指定起始位置出发，按照国际象棋规则移动（走"日"字），尝试访问棋盘上的每个格子恰好一次。以下是详细解释：

## 回溯算法
### 代码功能说明
1. **问题类型**：寻找骑士巡游路径（哈密顿路径问题）
2. **输入**：
   - 棋盘大小 n
   - 起始位置 (sr, sc)
3. **输出**：
   - 找到完整路径：输出 "success"
   - 找不到路径：输出 "fail"

### 核心算法：回溯法 + Warnsdorff 启发式优化
```python
def backtrack(x, y, step):
    if step == n * n:  # 终止条件：已访问所有格子
        return True
    
    # 1. 计算所有可行移动
    moves = []
    for dx, dy in directions:  # 8个移动方向
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
            # 2. Warnsdorff规则：计算下一步的可行移动数
            count = 0
            for dx2, dy2 in directions:
                next_x, next_y = nx + dx2, ny + dy2
                if 0 <= next_x < n and 0 <= next_y < n and not visited[next_x][next_y]:
                    count += 1
            moves.append((nx, ny, count))
    
    # 3. 按可行移动数升序排序（关键优化！）
    moves.sort(key=lambda x: x[2])
    
    # 4. 尝试每个移动
    for nx, ny, _ in moves:
        visited[nx][ny] = True
        if backtrack(nx, ny, step + 1):  # 递归探索
            return True
        visited[nx][ny] = False  # 回溯
    
    return False
```

### 关键优化：Warnsdorff 规则
1. **核心思想**：优先选择**下一步选择最少的格子**
   - 避免进入"死胡同"
   - 大幅减少回溯次数
2. **实现步骤**：
   - 对每个候选移动，计算从该位置出发的下一步选择数 (`count`)
   - 按 `count` 升序排序（先尝试可能造成死胡同的位置）

### 执行流程
1. **初始化**：
   ```python
   visited = [[False] * n for _ in range(n)]  # 访问标记矩阵
   visited[sr][sc] = True  # 标记起始位置
   ```
2. **开始回溯搜索**：
   ```python
   if backtrack(sr, sc, 1):  # 从起始位置开始（step=1）
       print("success")
   else:
       print("fail")
   ```

### 示例执行过程（3×3棋盘）
```
输入: n=3, 起始位置(0,0)
移动尝试顺序：
(0,0) → (2,1) → (1,0) → (0,2) → ...（回溯）
最终输出: "fail"（3×3棋盘无解）
```


## KMP
KMP算法（Knuth-Morris-Pratt算法）主要解决字符串匹配问题，即在一个主文本串（Text）中高效查找特定模式串（Pattern）的出现位置。例如在文本 "ABABABABC" 中查找模式 "ABABC"。
### 模板
```python
def kmp_search(text: str, pattern: str) -> int:
    if not pattern: return 0
    next_arr = build_next(pattern)  # 构建 next 数组
    i, j = 0, 0                    # i: 主串指针, j: 子串指针
    while i < len(text):
        if j == -1 or text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):   # 匹配成功
                return i - j        # 返回起始位置
        else:
            j = next_arr[j]         # 失配时跳转
    return -1                       # 未找到

def build_next(s: str) -> list[int]:
    next_arr = [-1] * len(s)
    i, j = 0, -1
    while i < len(s) - 1:
        if j == -1 or s[i] == s[j]:
            i += 1
            j += 1
            next_arr[i] = j
        else:
            j = next_arr[j]
    return next_arr

# 示例
text = "ABABABABC"
pattern = "ABABC"
print(kmp_search(text, pattern))  # 输出: 2（子串从索引 2 开始："ABABC"）
```

以下是一个**Python 链表操作的完整代码模板**，涵盖链表定义、常见操作（创建、遍历、插入、删除、反转等）和经典问题（查找中间节点、检测环、合并有序链表等），并附有详细注释：

---

## 链表
### **1. 链表节点定义**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val     # 节点存储的数据
        self.next = next   # 指向下一个节点的指针
```

---

### **2. 创建链表（尾插法）**
```python
def create_linked_list(values: List[int]) -> ListNode:
    dummy = ListNode(0)  # 虚拟头节点
    curr = dummy
    for val in values:
        curr.next = ListNode(val)  # 创建新节点并连接
        curr = curr.next
    return dummy.next  # 返回真实头节点
```

---

### **3. 遍历链表**
```python
def print_list(head: ListNode) -> None:
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")
```

---

### **4. 插入节点**
#### **(1) 头插法插入节点**
```python
def insert_at_head(head: ListNode, val: int) -> ListNode:
    new_node = ListNode(val)
    new_node.next = head
    return new_node
```

#### **(2) 尾插法插入节点**
```python
def insert_at_tail(head: ListNode, val: int) -> ListNode:
    if not head:
        return ListNode(val)
    curr = head
    while curr.next:
        curr = curr.next
    curr.next = ListNode(val)
    return head
```

#### **(3) 插入到指定位置**
```python
def insert_at_position(head: ListNode, val: int, pos: int) -> ListNode:
    if pos == 0:
        return insert_at_head(head, val)
    curr = head
    for _ in range(pos - 1):
        if not curr:
            break
        curr = curr.next
    if not curr:
        return head
    new_node = ListNode(val)
    new_node.next = curr.next
    curr.next = new_node
    return head
```

---

### **5. 删除节点**
#### **(1) 删除头节点**
```python
def delete_head(head: ListNode) -> ListNode:
    if not head:
        return None
    return head.next
```

#### **(2) 删除尾节点**
```python
def delete_tail(head: ListNode) -> ListNode:
    if not head or not head.next:
        return None
    curr = head
    while curr.next.next:
        curr = curr.next
    curr.next = None
    return head
```

#### **(3) 删除指定值的节点**
```python
def delete_val(head: ListNode, val: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    curr = dummy
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return dummy.next
```

---

### **6. 反转链表**
#### **(1) 迭代法反转**
```python
def reverse_list(head: ListNode) -> ListNode:
    prev = None
    curr = head
    while curr:
        next_temp = curr.next  # 临时保存下一个节点
        curr.next = prev       # 反转当前节点的指针
        prev = curr            # 移动 prev 和 curr
        curr = next_temp
    return prev  # 返回新的头节点
```

#### **(2) 递归法反转**
```python
def reverse_list_recursive(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

---

### **7. 查找中间节点（快慢指针）**
```python
def find_middle(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # slow 指向中间节点
```

---

### **8. 检测链表是否有环**
```python
def has_cycle(head: ListNode) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True  # 有环
    return False  # 无环
```

---

### **9. 合并两个有序链表**
```python
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2
    return dummy.next
```

---

### **10. 删除重复节点（有序链表）**
```python
def delete_duplicates(head: ListNode) -> ListNode:
    curr = head
    while curr and curr.next:
        if curr.val == curr.next.val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
```

---


---




## 栈

在 Python 中，**栈**（Stack）是一种遵循 **后进先出**（LIFO, Last In First Out）原则的数据结构。通常可以通过 **列表**（`list`）来实现栈的基本操作。

---

### **1. 栈的基本操作**
| 操作         | 描述                           | Python 实现                   |
|--------------|--------------------------------|-------------------------------|
| **入栈**     | 将元素添加到栈顶               | `stack.append(item)`          |
| **出栈**     | 移除并返回栈顶元素             | `stack.pop()`                 |
| **查看栈顶** | 返回栈顶元素但不移除它         | `stack[-1]`                   |
| **判断空**   | 检查栈是否为空                 | `len(stack) == 0`             |
| **栈的大小** | 返回栈中元素的数量             | `len(stack)`                  |

---

### **2. 使用列表实现栈**
#### **示例代码**
```python
# 初始化栈
stack = []

# 入栈操作
stack.append(1)  # 栈: [1]
stack.append(2)  # 栈: [1, 2]
stack.append(3)  # 栈: [1, 2, 3]

# 出栈操作
print(stack.pop())  # 输出: 3，栈变为 [1, 2]
print(stack.pop())  # 输出: 2，栈变为 [1]

# 查看栈顶元素
print(stack[-1])  # 输出: 1

# 判断栈是否为空
print(len(stack) == 0)  # 输出: False

# 获取栈的大小
print(len(stack))  # 输出: 1
```

---

### **3. 封装为栈类（自定义实现）**
如果需要更规范的接口，可以封装一个 `Stack` 类：

```python
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.items[-1]

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)

# 使用示例
s = Stack()
s.push(1)
s.push(2)
s.push(3)
print(s.pop())    # 输出: 3
print(s.peek())   # 输出: 2
print(s.size())   # 输出: 2
print(s)          # 输出: [1, 2]
```

---

### **4. 其他实现方式**
#### **4.1 使用 `collections.deque`**
`deque` 是双向队列，也可以高效地实现栈：
```python
from collections import deque

stack = deque()
stack.append(1)  # 入栈
stack.append(2)
print(stack.pop())  # 出栈，输出: 2
```

#### **4.2 使用 `queue.LifoQueue`**
`LifoQueue` 是线程安全的栈实现：
```python
from queue import LifoQueue

stack = LifoQueue()
stack.put(1)       # 入栈
stack.put(2)
print(stack.get()) # 出栈，输出: 2
```

---

### **5. 栈的典型应用场景**
1. **括号匹配**：通过栈检查括号是否闭合。
2. **表达式求值**：用于中缀表达式转后缀表达式（逆波兰表达式）。
3. **函数调用栈**：模拟程序执行时的函数调用过程。
4. **字符串逆序**：将字符串逐个字符入栈后出栈，得到逆序结果。

---

### **6. 注意事项**
- **异常处理**：在出栈或查看栈顶元素时，需检查栈是否为空，避免抛出 `IndexError`。
- **性能优化**：使用列表的 `append()` 和 `pop()` 操作时间复杂度为 **O(1)**，效率较高。

---

### **7. 示例：字符串逆序**
```python
def reverse_string(s):
    stack = []
    for char in s:
        stack.append(char)
    reversed_str = ""
    while stack:
        reversed_str += stack.pop()
    return reversed_str

print(reverse_string("hello"))  # 输出: "olleh"
```

---

## **Kahn 算法**

是用于 **有向无环图（DAG）** 的 **拓扑排序算法**，通过不断移除入度为 0 的节点生成拓扑序列。以下是其核心原理和实现细节：

---

### **算法核心思想**
1. **入度表 (In-degree Table)**  
   记录每个节点的**入度**（指向该节点的边数）。
2. **队列/堆栈**  
   维护当前所有**入度为 0 的节点**，决定处理顺序（队列实现广度优先，堆栈实现深度优先）。
3. **逐步移除节点**  
   每次从队列中取出一个节点，将其输出到拓扑序列，并**减少其邻接节点的入度**。若邻接节点入度变为 0，则加入队列。

---


### **算法步骤**
1. **初始化**  
   - 计算所有节点的入度，存入 `in_degree` 表。
   - 将入度为 0 的节点加入队列。
   
2. **循环处理**  
   ```python
   while 队列不为空:
       u = 队列.pop()         # 取出一个入度为 0 的节点
       拓扑序列.append(u)     # 加入结果
       for v in u 的邻接节点:
           in_degree[v] -= 1 # 减少邻接节点的入度
           if in_degree[v] == 0:
               队列.append(v) # 若入度归零，加入队列
   ```

3. **检测环路**  
   - 若最终拓扑序列的节点数 **≠ 总节点数** → 图中存在环（无法拓扑排序）。

---

### **示例演示**
**有向图示例**：  
节点关系：`A → B → C`，`A → D → C`

1. **初始化**  
   - 入度表：`A:0`, `B:1`, `C:2`, `D:1`
   - 队列：`[A]`

2. **执行过程**  
   - 取出 `A`，拓扑序 `[A]`，更新 `B` 和 `D` 的入度为 0，队列变为 `[B, D]`
   - 取出 `B`，拓扑序 `[A, B]`，更新 `C` 的入度为 1，队列 `[D]`
   - 取出 `D`，拓扑序 `[A, B, D]`，更新 `C` 的入度为 0，队列 `[C]`
   - 取出 `C`，拓扑序 `[A, B, D, C]`，结束

3. **结果**  
   拓扑序列：`A → B → D → C`（或 `A → D → B → C`，取决于队列的取出顺序）

---

### **代码实现（Python）**
```python
from collections import deque

def topological_sort_kahn(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != len(graph):
        return None  # 存在环路
    return topo_order

# 示例图
graph = {
    'A': ['B', 'D'],
    'B': ['C'],
    'D': ['C'],
    'C': []
}

print(topological_sort_kahn(graph))  # 输出：['A', 'B', 'D', 'C']
```

---

### **算法特性**
| 特性                  | 说明                                                                 |
|-----------------------|----------------------------------------------------------------------|
| **时间复杂度**        | O(V + E)（V 为节点数，E 为边数）                                       |
| **空间复杂度**        | O(V)（队列和入度表存储）                                               |
| **适用场景**          | 任务调度、依赖解析（如编译顺序）、课程安排等                           |
| **环路检测**          | 若最终拓扑序列不包含所有节点 → 图中存在环                               |
| **结果多样性**        | 同一 DAG 可能有多个合法拓扑序，具体结果取决于队列的节点取出顺序          |

---

### **应用场景**
1. **任务调度**  
   确定任务的执行顺序，确保前置任务先完成。
2. **编译顺序**  
   解决源代码文件或库的依赖关系。
3. **课程安排**  
   根据课程先修关系生成合理的学习顺序。
4. **数据流分析**  
   优化计算顺序以减少中间存储。

---

### **对比其他拓扑排序算法**
| 算法          | 优点                          | 缺点                          |
|---------------|-------------------------------|-------------------------------|
| **Kahn**      | 直观，易于实现环路检测        | 需要维护入度表和队列           |
| **DFS 回溯**  | 无需预处理入度                | 需要递归栈，可能不适用于大规模图 |


## **Dijkstra算法**

Dijkstra算法是计算机科学中用于解决**单源最短路径问题**的经典算法，由荷兰计算机科学家艾兹赫尔·戴克斯特拉（Edsger W. Dijkstra）于1956年提出。它适用于**没有负权边**的有向图或无向图，能够高效地计算出从一个源点到图中所有其他节点的最短路径。

---

### **核心思想**
Dijkstra算法基于**贪心策略**，通过逐步扩展当前已知的最短路径集合，最终覆盖所有节点。其核心思想是：
1. **初始化**：将源点的最短距离设为0，其他节点的最短距离初始化为无穷大（∞）。
2. **选择最短距离节点**：从未访问的节点中选择当前距离源点最近的节点作为“中间桥梁”。
3. **更新邻居距离**：通过该节点更新其邻居节点的最短距离。
4. **重复扩展**：直到所有节点都被访问或目标节点的最短路径被确定。

---

### **算法步骤**
以图中的邻接矩阵为例，假设源点为 $ A $，目标是计算 $ A $ 到所有其他节点的最短路径：
1. **初始化**：
   - 距离数组 `dist[]`：记录源点到每个节点的最短距离。初始时，`dist[A] = 0`，其他节点设为 `∞`。
   - 未访问集合 `unvisited`：包含所有节点。
   - 前驱节点数组 `prev[]`：记录每个节点的最短路径前驱节点（可选）。

2. **主循环**：
   - **选择当前最短距离节点**：从 `unvisited` 中选择 `dist[u]` 最小的节点 $ u $。
   - **标记已访问**：将 $ u $ 从未访问集合中移除。
   - **更新邻居距离**：对 $ u $ 的所有邻居 $ v $，如果通过 $ u $ 到 $ v $ 的路径更短（即 `dist[u] + weight(u, v) < dist[v]`），则更新 `dist[v]` 和 `prev[v]`。

3. **终止条件**：
   - 所有节点被访问，或目标节点的最短路径被确定。

---

### **示例演示**
假设图的邻接矩阵如下（节点为 $ A, B, C, D $）：
```
   A  B  C  D
A  0  2  ∞  6
B  2  0  3  2
C  ∞  3  0  2
D  6  2  2  0
```
- **初始化**：`dist = [0, ∞, ∞, ∞]`，`unvisited = {A, B, C, D}`。
- **第一轮**：选择距离最小的节点 $ A $（距离为0），更新其邻居 $ B $（2）和 $ D $（6）。
- **第二轮**：选择 $ B $（当前最小距离为2），更新其邻居 $ C $（2+3=5）和 $ D $（2+2=4）。
- **第三轮**：选择 $ D $（当前最小距离为4），更新其邻居 $ C $（4+2=6，不更新）。
- **第四轮**：选择 $ C $（当前最小距离为5），无更优路径。
- **结果**：`dist = [0, 2, 5, 4]`。

---

### **时间复杂度**
- **基础实现**：使用数组存储距离和未访问节点，时间复杂度为 $ O(V^2) $，其中 $ V $ 是节点数。
- **优化实现**：使用**优先队列（最小堆）**优化选择最短距离节点的操作，时间复杂度为 $ O((V + E) \log V) $，其中 $ E $ 是边数。
- **进一步优化**：使用**斐波那契堆**，时间复杂度可降至 $ O(V \log V + E) $。

---

### **算法限制**
1. **不能处理负权边**：如果图中存在负权边，Dijkstra算法可能无法正确计算最短路径（需改用 Bellman-Ford 算法）。
2. **动态图不适用**：如果图的结构（边权重）频繁变化，需重新运行算法。

---

### Dijkstra算法的**最小堆优化**（也称为**优先队列优化**）

是通过使用**最小堆**（或**优先队列**）来高效维护“未访问节点集合”，从而快速找到当前距离源点最近的节点。这种方法显著提升了算法的效率，尤其适用于边数较少的稀疏图。

---

### **最小堆优化的核心思想**
1. **最小堆的作用**：
   - 最小堆是一种特殊的树形数据结构，堆顶元素始终是当前堆中最小的。
   - 在Dijkstra算法中，最小堆用于存储“待处理的节点及其当前距离源点的最短距离”，堆顶元素即为当前距离源点最近的节点。

2. **优化目标**：
   - 传统Dijkstra算法中，每次需要遍历所有节点以找到距离源点最近的节点（时间复杂度为 $ O(V) $）。堆优化后，这一操作的时间复杂度降至 $ O(\log V) $。
   - 总体时间复杂度从 $ O(V^2) $ 降低到 $ O((V + E) \log V) $，其中 $ V $ 是节点数，$ E $ 是边数。

3. **堆的动态维护**：
   - 当某个节点的距离被更新时，将其新距离和节点重新加入堆中。
   - 即使堆中存在旧的、无效的条目（如节点已被处理），它们也会被忽略（通过标记节点是否已访问）。

---

### **算法步骤详解**
1. **初始化**：
   - **距离数组** `dist[]`：记录源点到每个节点的最短距离，初始化为无穷大（`∞`），源点距离为0。
   - **访问标记数组** `visited[]`：标记节点是否已被处理。
   - **最小堆**：将源点（距离0）加入堆。

2. **主循环**：
   - **弹出堆顶元素**：取出当前距离源点最近的节点 $ u $。
   - **跳过已访问节点**：若 $ u $ 已被处理，直接跳过。
   - **标记已访问**：将 $ u $ 标记为已访问。
   - **更新邻居距离**：遍历 $ u $ 的所有邻接节点 $ v $：
     - 若通过 $ u $ 到 $ v $ 的路径更短（即 `dist[u] + weight(u, v) < dist[v]`），则更新 `dist[v]`，并将 $ v $ 和新的距离加入堆中。

3. **终止条件**：
   - 堆为空时，所有节点的最短路径已确定。

---

### **代码示例（Python）**
```python
import heapq

def dijkstra_heap(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n
    heap = [(0, start)]  # (distance, node)

    while heap:
        current_dist, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, weight in graph[u]:
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                heapq.heappush(heap, (dist[v], v))
    return dist

# 示例图（邻接表）
graph = [
    [(1, 2), (3, 6)],  # A (index 0)
    [(0, 2), (2, 3), (3, 2)],  # B (index 1)
    [(1, 3), (3, 2)],  # C (index 2)
    [(0, 6), (1, 2), (2, 2)]  # D (index 3)
]
print(dijkstra_heap(graph, 0))  # 输出: [0, 2, 5, 4]
```

---


### **关键点解析**
1. **堆的动态性**：
   - 堆中可能包含重复的节点（如节点 $ v $ 的旧距离和新距离），但通过 `visited[]` 标记，旧条目会被自动忽略。
   - 例如，当节点 $ v $ 的距离被更新后，旧的 `dist[v]` 仍可能在堆中，但一旦 `v` 被标记为已访问，后续弹出的旧条目会被跳过。

2. **邻接表 vs 邻接矩阵**：
   - **邻接表**：适合稀疏图（边数远小于 $ V^2 $），节省空间。
   - **邻接矩阵**：适合稠密图，但空间复杂度为 $ O(V^2) $。

3. **时间复杂度分析**：
   - 每个边最多被处理一次，每次堆操作的时间复杂度为 $ O(\log V) $。
   - 总体时间复杂度为 $ O((V + E) \log V) $。

---

### **常见问题与解决方案**
1. **堆中存在无效条目怎么办**？
   - 通过 `visited[]` 标记已访问的节点，确保每个节点仅被处理一次。

2. **如何处理负权边？**
   - 堆优化的Dijkstra算法**不适用于负权边**。需改用 **Bellman-Ford算法** 或 **SPFA算法**。

3. **如何获取最短路径本身？**
   - 维护一个 `prev[]` 数组记录每个节点的前驱节点，最后通过回溯 `prev[]` 构建路径。

---
### 例题
#### 题目
##### 描述
N个以 1 ... N 标号的城市通过单向的道路相连:。每条道路包含两个参数：道路的长度和需要为该路付的通行费（以金币的数目来表示）

Bob and Alice 过去住在城市 1.在注意到Alice在他们过去喜欢玩的纸牌游戏中作弊后，Bob和她分手了，并且决定搬到城市N。他希望能够尽可能快的到那，但是他囊中羞涩。我们希望能够帮助Bob找到从1到N最短的路径，前提是他能够付的起通行费。

##### 输入
第一行包含一个整数K, 0 <= K <= 10000, 代表Bob能够在他路上花费的最大的金币数。第二行包含整数N， 2 <= N <= 100, 指城市的数目。第三行包含整数R, 1 <= R <= 10000, 指路的数目.
接下来的R行，每行具体指定几个整数S, D, L 和 T来说明关于道路的一些情况，这些整数之间通过空格间隔:
S is 道路起始城市, 1 <= S <= N
D is 道路终点城市, 1 <= D <= N
L is 道路长度, 1 <= L <= 100
T is 通行费 (以金币数量形式度量), 0 <= T <=100
注意不同的道路可能有相同的起点和终点。
##### 输出
输入结果应该只包括一行，即从城市1到城市N所需要的最小的路径长度（花费不能超过K个金币）。如果这样的路径不存在，结果应该输出-1。
##### 样例输入
5
6
7
1 2 2 3
2 4 3 3
3 4 2 4
1 3 4 1
4 6 2 1
3 5 2 0
5 4 3 2
##### 样例输出
11

#### 代码
```python
import heapq

def dijkstra_with_cost(graph, start, K):
    n = len(graph)
    INF = float('inf')
    # dist[u][c] 表示从起点到城市u，花费c金币的最小路径长度
    dist = [[INF] * (K + 1) for _ in range(n)]
    dist[start][0] = 0  # 初始：从起点出发，花费0金币，路径长度为0
    heap = [(0, start, 0)]  # (distance, node, cost)

    while heap:
        d, u, c = heapq.heappop(heap)
        if d > dist[u][c]:
            continue  # 已经找到更优的路径，跳过
        for v, l, t in graph[u]:
            new_c = c + t
            if new_c > K:
                continue  # 超出预算，跳过
            new_d = d + l
            if new_d < dist[v][new_c]:
                dist[v][new_c] = new_d
                heapq.heappush(heap, (new_d, v, new_c))

    # 在所有满足通行费限制的路径中，找出最短路径长度
    min_len = min(dist[n - 1][c] for c in range(K + 1) if dist[n - 1][c] < INF)
    return min_len if min_len < INF else -1

# 读取输入
K = int(input())
N = int(input())
R = int(input())
graph = [[] for _ in range(N)]

for _ in range(R):
    S, D, L, T = map(int, input().split())
    graph[S - 1].append((D - 1, L, T))  # 存储邻接表

# 调用算法并输出结果
print(dijkstra_with_cost(graph, 0, K))
```



---
## 树
### **一、树（Tree）**
#### **1. 基本定义**
- **树** 是一种非线性数据结构，由节点（Node）和边（Edge）组成，满足以下条件：
  - 有且仅有一个根节点（Root），无父节点。
  - 除根节点外，每个节点有且仅有一个父节点。
  - 所有节点通过边连接，形成层次结构，无环路。

#### **2. 关键术语**
- **根节点**：树的顶端节点，无父节点。
- **子节点**：某个节点的直接下级节点。
- **父节点**：某个节点的直接上级节点。
- **叶子节点**：没有子节点的节点。
- **子树**：以某个节点为根的子树，包含其所有后代节点。
- **深度**：从根节点到当前节点的路径长度。
- **高度**：从当前节点到最远叶子节点的路径长度。
---

#### 例题：森林的带度数层次序列存储
#### 描述
对于树和森林等非线性结构，我们往往需要将其序列化以便存储。有一种树的存储方式称为带度数的层次序列。我们可以通过层次遍历的方式将森林序列转化为多个带度数的层次序列。

例如对于以下森林：
两棵树的层次遍历序列分别为：C E F G K H J / D X I
每个结点对应的度数为：3 3 0 0 0 0 0 / 2 0 0
我们将以上序列存储起来，就可以在以后的应用中恢复这个森林。在存储中，我们可以将第一棵树表示为C 3 E 3 F 0 G 0 K 0 H 0 J 0，第二棵树表示为D 2 X 0 I 0。
现在有一些通过带度数的层次遍历序列存储的森林数据，为了能够对这些数据进行进一步处理，首先需要恢复他们。

#### 输入
输入数据的第一行包括一个正整数n，表示森林中非空的树的数目。
随后的 n 行，每行给出一棵树的带度数的层次序列。
树的节点名称为A-Z的单个大写字母。
#### 输出
输出包括一行，输出对应森林的后根遍历序列。
#### 样例输入
2
C 3 E 3 F 0 G 0 K 0 H 0 J 0
D 2 X 0 I 0
#### 样例输出
K H J E F G C X I D

#### 代码示例
```python
from collections import deque
from typing import List, Optional

# 定义树节点类，每个节点包含一个值和子节点列表
class TreeNode:
    def __init__(self, val: str):
        self.val = val  # 节点的值（字符串类型）
        self.child = []  # 子节点列表（多叉树）

    # 添加子节点的方法
    def add_child(self, child: 'TreeNode'):# 在TreeNode类中使用TreeNode类，需要引号
        self.child.append(child)

# 解题类，包含构建树和后序遍历的方法
class Solution:
    # 构建树的函数
    # 输入参数 s 是一个字符串列表，格式为：
    def build_tree(self, s: list) -> Optional[TreeNode]:
        if not s:
            return None  # 空输入返回空

        # 创建根节点
        root = TreeNode(s[0])
        # 使用队列保存待处理的节点及其度数
        queue = deque([(root, int(s[1]))])
        i = 1  # 当前处理的位置索引

        # 处理所有节点
        while i < len(s) // 2:
            node, degree = queue.popleft()  # 取出当前节点及其度数
            for _ in range(degree):  # 根据度数添加子节点
                child = TreeNode(s[2 * i])  # 子节点的值
                node.add_child(child)  # 添加到当前节点的子节点列表
                queue.append((child, int(s[2 * i + 1])))  # 将子节点加入队列
                i += 1  # 移动到下一个节点信息

        return root  # 返回构建完成的根节点

    # 后序遍历函数（多叉树）
    # 输入参数 node 是树的根节点
    def trans_order(self, node: Optional[TreeNode]) -> List[str]:
        if node is None:
            return []  # 空节点返回空列表

        result = []
        # 递归处理所有子节点
        for child in node.child:
            result.extend(self.trans_order(child))

        # 当前节点在所有子节点处理完后添加到结果中
        result.append(node.val)
        return result


# 主程序入口
if __name__ == '__main__':
    n = int(input())  # 输入测试用例数量
    for _ in range(n):
        s = input().split()  # 读取每个测试用例的字符串列表
        # 构建树并获取后序遍历结果
        root = Solution().build_tree(s)
        result = Solution().trans_order(root)
        # 输出结果（空格分隔）
        print(' '.join(result), end=' ')
```



### **二、二叉树（Binary Tree）**
#### **1. 基本定义**
- **二叉树** 是每个节点最多有 **两个子节点** 的树，分别称为 **左子节点** 和 **右子节点**。
- **特点**：
  - 子节点有明确的左右顺序。
  - 可以是空树（无节点）。

#### **2. 常见类型**
| **类型**          | **定义**                                                                 | **示例用途**               |
|--------------------|-------------------------------------------------------------------------|---------------------------|
| **满二叉树**       | 每个节点要么是叶子，要么有两个子节点。                                     | 堆结构（Heap）             |
| **完全二叉树**     | 除最后一层外，其他层节点全满，且最后一层节点左对齐。                       | 优先队列、堆排序           |
| **二叉搜索树**     | 左子树所有节点的值 < 根节点值 < 右子树所有节点的值。                        | 快速查找、插入、删除（时间复杂度 O(log n)） |
| **平衡二叉树**     | 左右子树高度差不超过 1（如 AVL 树、红黑树）。                               | 保证操作效率稳定           |
| **哈夫曼树**       | 带权路径长度最短的二叉树，用于数据压缩。                                   | 文件压缩（如 ZIP）         |

#### **3. 二叉树的遍历方式**
- **深度优先遍历（DFS）**：
  - **前序遍历**：根 → 左 → 右（用于复制树结构）。
  - **中序遍历**：左 → 根 → 右（二叉搜索树输出有序序列）。
  - **后序遍历**：左 → 右 → 根（用于释放树内存）。
- **广度优先遍历（BFS）**：按层次逐层访问节点（队列实现）。

##### 例题：根据二叉树前中序序列建树
###### 描述
假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。

给定二叉树的前序遍历序列和中序遍历序列(长度均不超过26)，请输出该二叉树的后序遍历序列

###### 输入
多组数据
每组数据2行，第一行是前序遍历序列，第二行是中序遍历序列
###### 输出
对每组序列建树，输出该树的后序遍历序列
###### 样例输入
DURPA
RUDPA
XTCNB
CTBNX
###### 样例输出
RUAPD
CBNTX
###### 示例代码
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        
def buildtree(pre_or,in_or):
    if not pre_or :
        return None
    root_val=pre_or[0]
    root=TreeNode(root_val)
    root_sit=in_or.index(root_val)
    left_in=in_or[:root_sit]
    right_in=in_or[root_sit+1:]
    left_pre=pre_or[1:len(left_in)+1]
    right_pre=pre_or[len(left_in)+1:]
    root.left=buildtree(left_pre,left_in)
    root.right=buildtree(right_pre,right_in)
    
    return root

def print_post_order(root):
    if root is None:
        return None
    if root.left is not None :
        print_post_order(root.left)
    if root.right is not None :
        print_post_order(root.right)
        
    print(root.val,end='')
          
while True:
    try:
        pre_order=input()
        in_order=input()
        print_post_order(buildtree(pre_order,in_order))
        print()
    except EOFError:
        break
```
#### **4. 二叉树的存储结构**
- **链式存储**：通过节点对象和指针（或引用）表示父子关系。
  ```python
  class TreeNode:
      def __init__(self, value):
          self.val = value
          self.left = None
          self.right = None
  ```
- **顺序存储**：用数组表示完全二叉树，下标关系为：
  - 父节点索引 `i` → 左子节点 `2i+1`，右子节点 `2i+2`。

---

### **三、实际应用示例**
#### **1. 二叉搜索树（BST）**
- **插入**：递归比较值大小，找到合适位置插入。
- **查找**：类似二分查找，时间复杂度 O(log n)。
- **删除**：分三种情况（无子节点、有一个子节点、有两个子节点）。

#### **2. 哈夫曼编码**
- **步骤**：
  1. 统计字符频率，构建哈夫曼树。
  2. 左路径标记 0，右路径标记 1，生成字符编码。
  3. 压缩时用编码替代原字符。

#### **3. 堆（完全二叉树）**
- **最小堆**：父节点值 ≤ 子节点值。
- **最大堆**：父节点值 ≥ 子节点值。
- **应用**：堆排序、Top K 问题、优先队列。

---

### 完全二叉树例题
#### 题目
##### 描述
探险家小B发现了一颗宝藏二叉树。这棵树的树根为Root，除了Root节点之外，每个节点均只有一个父节点，因此形成了一颗二叉树。宝藏二叉树的每个节点都有宝藏，每个宝藏具有相应的价值。小B希望摘取这些宝藏，使自己的收益最大。可是，宝藏二叉树有一个奇怪的性质，在摘取宝藏的时候，如果两个节点之间有边，那么最多只能摘取其中一个节点上的宝藏，如果因为贪婪而把两个节点上的宝藏都摘取，二叉树就会立即消失，丧失所有奖励。为此，小B求助于你，希望你能给出，小B在不使宝藏二叉树消失的前提下，能够获得宝藏的最大价值。

为了简化题目，规定宝藏二叉树均为完全二叉树，树中节点如图所示自上而下，自左向右，从1-N编号。

##### 输入
输入分为两行
第一行为一个整数N，代表二叉树中节点的个数。
第二行为一个N个非负整数。第i个数代表二叉树中编号为i的节点上的宝藏价值。
##### 输出
输出为一个整数，代表小B的最大收益。
##### 样例输入
6
3 4 5 1 3 1
##### 样例输出
9

#### 代码
```python
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    vals = list(map(int, data[1:]))

    # dp[i][0]: 不选第i个节点的最大收益
    # dp[i][1]: 选第i个节点的最大收益
    dp = [[0, 0] for _ in range(n + 2)]  # 为方便处理，多开2个位置

    for i in range(n, 0, -1):  # 从下往上处理
        val = vals[i - 1]
        left = 2 * i
        right = 2 * i + 1

        # 选当前节点
        select = val
        if left <= n:
            select += dp[left][0]
        if right <= n:
            select += dp[right][0]
        dp[i][1] = select

        # 不选当前节点
        not_select = 0
        max_left = max(dp[left][0], dp[left][1]) if left <= n else 0
        max_right = max(dp[right][0], dp[right][1]) if right <= n else 0
        dp[i][0] = max_left + max_right

    print(max(dp[1][0], dp[1][1]))

if __name__ == "__main__":
    main()
```

## 霍夫曼编码树（Huffman Coding Tree）
### 核心原理
1. **变长编码**：不同字符有不同的编码长度
2. **前缀编码**：任何字符的编码都不是其他字符编码的前缀
3. **贪心策略**：每次合并频率最小的两个节点
### 重要概念
- **频率**：字符在文本中出现的次数
- **叶子节点**：代表原始字符
- **内部节点**：代表合并后的字符集
- **路径编码**：从根到叶子的路径（左0右1）形成字符编码

### 构建霍夫曼树的步骤

1. **统计频率**：计算文本中每个字符的出现频率
2. **创建节点**：为每个字符创建带权重的节点
3. **构建优先队列**：将所有节点放入最小堆（按频率排序）
4. **合并节点**：
   - 取出两个频率最小的节点
   - 创建新父节点（权重=子节点权重和）
   - 将新节点放回堆中
5. **重复合并**：直到堆中只剩一个节点（根节点）
6. **生成编码**：从根节点遍历树，左分支为0，右分支为1

### 示例演示

#### 输入文本："BCCABBDDAECCBBA"

1. **统计频率**：
   ```
   A: 3, B: 5, C: 4, D: 2, E: 1
   ```

2. **构建过程**：
   ```
   初始节点： [A:3], [B:5], [C:4], [D:2], [E:1]
   
   步骤1：合并 E(1) 和 D(2) -> 新节点(3)
       剩余：A(3), C(4), B(5), [ED](3)
   
   步骤2：合并 A(3) 和 [ED](3) -> 新节点(6)
       剩余：C(4), B(5), [AED](6)
   
   步骤3：合并 C(4) 和 B(5) -> 新节点(9)
       剩余：[AED](6), [CB](9)
   
   步骤4：合并 [AED](6) 和 [CB](9) -> 根节点(15)
   ```

3. **最终树结构**：
   ```
         (15)
        /    \
      (6)    (9)
     /  \    /  \
    A   (3) C    B
        / \
       E   D
   ```

4. **生成编码**：
   ```
   A: 00
   E: 010
   D: 011
   C: 10
   B: 11
   ```

5. **编码结果**：
   原始文本：BCCABBDDAECCBBA
   编码后：11101010100011110110110001010101100

### 代码实现

```python
import heapq
from collections import defaultdict

class Node:
    """霍夫曼树节点类"""
    def __init__(self, char, freq):
        self.char = char  # 字符（内部节点为None）
        self.freq = freq  # 频率
        self.left = None
        self.right = None
    
    # 定义比较运算符用于堆排序
    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_table(data):
    """构建字符频率表"""
    freq = defaultdict(int)
    for char in data:
        freq[char] += 1
    return freq

def build_huffman_tree(freq_table):
    """构建霍夫曼树"""
    # 创建优先队列（最小堆）
    heap = []
    for char, freq in freq_table.items():
        heapq.heappush(heap, Node(char, freq))
    
    # 合并节点直到只剩一个
    while len(heap) > 1:
        # 取出两个最小节点
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # 创建新节点（内部节点）
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        # 将新节点放回堆中
        heapq.heappush(heap, merged)
    
    return heap[0]  # 返回根节点

def generate_codes(root, current_code="", codes={}):
    """递归生成霍夫曼编码"""
    if root is None:
        return
    
    # 叶子节点：存储字符编码
    if root.char is not None:
        codes[root.char] = current_code
        return
    
    # 遍历左子树（添加0）
    generate_codes(root.left, current_code + "0", codes)
    # 遍历右子树（添加1）
    generate_codes(root.right, current_code + "1", codes)
    
    return codes

def huffman_encode(data):
    """霍夫曼编码完整流程"""
    if not data:
        return "", None
    
    # 1. 构建频率表
    freq_table = build_frequency_table(data)
    
    # 2. 构建霍夫曼树
    root = build_huffman_tree(freq_table)
    
    # 3. 生成编码字典
    codes = generate_codes(root)
    
    # 4. 编码数据
    encoded_data = "".join(codes[char] for char in data)
    
    return encoded_data, root

def huffman_decode(encoded_data, root):
    """霍夫曼解码"""
    if not encoded_data:
        return ""
    
    decoded_data = []
    current_node = root
    
    for bit in encoded_data:
        # 根据比特位选择路径
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        # 到达叶子节点
        if current_node.char is not None:
            decoded_data.append(current_node.char)
            current_node = root  # 重置到根节点
    
    return "".join(decoded_data)

# 测试示例
if __name__ == "__main__":
    text = "BCCABBDDAECCBBA"
    print("原始文本:", text)
    
    # 编码
    encoded, tree = huffman_encode(text)
    print("编码结果:", encoded)
    
    # 解码
    decoded = huffman_decode(encoded, tree)
    print("解码结果:", decoded)
    
    # 验证
    print("验证结果:", "成功" if text == decoded else "失败")
    
    # 输出编码表
    codes = generate_codes(tree)
    print("\n字符编码表:")
    for char, code in sorted(codes.items()):
        print(f"{char}: {code}")
```

### 输出结果

```
原始文本: BCCABBDDAECCBBA
编码结果: 11101010100011110110110001010101100
解码结果: BCCABBDDAECCBBA
验证结果: 成功

字符编码表:
A: 00
B: 11
C: 10
D: 011
E: 010
```
##### Trie（前缀树）类的详细解释与注释

以下是针对用户提供的 Trie 实现的逐行解释和注释，涵盖每个方法的功能、逻辑和设计意图。

---
## 前缀树trie
#### **1. TrieNode 类**
```python
class TrieNode:
    def __init__(self):
        self.p = 0  # 记录经过该节点的单词数量（pass count）
        self.e = 0  # 记录以该节点为结尾的单词数量（end count）
        self.next = [None] * 26  # 子节点数组，仅适用于小写字母 a-z
```

- **`p`（pass count）**：表示有多少个单词经过该节点。例如，插入 "app" 和 "apple" 后，每个节点的 `p` 会依次递增。
- **`e`（end count）**：表示有多少个单词以该节点为结尾。例如，插入 "app" 后，第三个 `p` 节点的 `e` 会增加 1。
- **`next`**：长度为 26 的数组，存储 26 个小写字母的子节点指针。若处理非字母字符，建议改用字典 `dict`。

---

#### **2. Trie 类的初始化**
```python
def __init__(self):
    self.root = self.TrieNode()
```

- 初始化 Trie 树的根节点。

---

#### **3. 插入方法 `insert`**
```python
def insert(self, word):
    node = self.root
    node.p += 1  # 根节点的 pass count 增加
    for i in range(len(word)):
        path = ord(word[i]) - ord('a')  # 将字符转换为索引
        if node.next[path] is None:
            node.next[path] = self.TrieNode()  # 创建新节点
        node = node.next[path]
        node.p += 1  # 当前节点的 pass count 增加
    node.e += 1  # 单词结尾节点的 end count 增加
    return
```

- **逻辑**：
  1. 从根节点开始，每经过一个节点，`p` 增加 1。
  2. 遍历单词的每个字符，动态创建子节点（如果不存在）。
  3. 最后一个字符对应的节点 `e` 增加 1。

- **示例**：插入 "apple" 后，路径上的每个节点的 `p` 会增加，最后一个 `e` 会增加 1。

---

#### **4. 搜索方法 `search`**
```python
def search(self, word):
    node = self.root
    for i in range(len(word)):
        path = ord(word[i]) - ord('a')
        if node.next[path] is None:
            return 0  # 路径不存在，单词未插入
        node = node.next[path]
    return node.e  # 返回以该节点结尾的单词数量
```

- **功能**：判断单词是否被插入过，并返回其出现次数。
- **返回值**：
  - `0`：单词未被插入或部分路径不存在。
  - `node.e`：单词被插入的次数。

---

#### **5. 前缀匹配方法 `startsWith`**
```python
def startsWith(self, word):
    node = self.root
    for i in range(len(word)):
        path = ord(word[i]) - ord('a')
        if node.next[path] is None:
            return 0  # 前缀不存在
        node = node.next[path]
    return node.p  # 返回以该前缀开头的单词数量
```

- **功能**：判断是否存在以给定前缀开头的单词。
- **返回值**：
  - `0`：前缀不存在。
  - `node.p`：以该前缀开头的单词数量。

---

#### **6. 删除方法 `delete`**
```python
def delete(self, word):
    if self.search(word) > 0:  # 确保单词存在
        node = self.root
        node.p -= 1  # 根节点 pass count 减少
        for i in range(len(word)):
            path = ord(word[i]) - ord('a')
            if node.next[path].p == 1:  # 如果子节点的 pass count 为 1
                node.next[path] = None  # 删除该子节点并提前返回
                return
            node = node.next[path]
            node.p -= 1  # 当前节点 pass count 减少
        node.e -= 1  # 最后一个节点的 end count 减少
    return
```

- **逻辑**：
  1. **存在性检查**：仅当 `search(word) > 0` 时执行删除。
  2. **路径回溯**：从根节点开始，逐层减少 `p`。
  3. **节点裁剪**：若某个子节点的 `p` 为 1，说明该子节点仅被当前单词使用，可安全删除，无需继续处理后续字符。
  4. **更新 `e`**：最后一个字符对应的节点 `e` 减少 1。

- **示例**：删除 "app" 后，若 "apple" 仍存在，则路径上的 `p` 会减少，但不会删除共享的节点。

---


### **注意事项**

1. **字符范围限制**：
   - 当前实现仅适用于小写字母 `a-z`。若需处理其他字符，应将 `next` 改为字典 `dict`。
   - 示例：`self.next = {}`，并替换 `ord(word[i]) - ord('a')` 为 `word[i]`。

2. **删除操作的裁剪逻辑**：
   - 删除时优先检查子节点的 `p` 是否为 1，确保仅删除唯一路径的节点，避免误删共享路径。
   - 若某个节点的 `p` 为 1，说明该节点仅被当前单词使用，可安全删除。

3. **时间复杂度**：
   - 所有操作的时间复杂度为 $ O(L) $，其中 $ L $ 是字符串的长度。
   - 空间复杂度取决于插入的字符串数量和字符集大小。

---

### **总结**

该 Trie 实现通过 `p` 和 `e` 两个属性，高效地统计了单词的插入、搜索和删除次数，并支持前缀匹配操作。删除逻辑通过裁剪无用节点优化了空间利用率，适用于需要频繁操作字符串集合的场景（如自动补全、拼写检查等）。
## 手搓最小堆

```python
class xiaoheap:
    def __init__(self):
        # 初始化堆，索引0位置不使用（置为0）
        self.heaplist = [0]  # 堆的存储列表（1-indexed）
        self.size = 0        # 堆中元素数量
    
    def percup(self, i):
        """上浮操作：将位置i的元素向上调整，维护堆结构"""
        while i // 2 > 0:  # 当i不是根节点时
            # 如果当前节点小于父节点，交换它们
            if self.heaplist[i] < self.heaplist[i // 2]:
                self.heaplist[i], self.heaplist[i // 2] = self.heaplist[i // 2], self.heaplist[i]
            i //= 2  # 移动到父节点位置
    
    def insert(self, i):
        """插入元素到堆中"""
        self.heaplist.append(i)  # 将新元素添加到堆末尾
        self.size += 1           # 堆大小增加
        self.percup(self.size)   # 对新元素进行上浮操作
    
    def minchild(self, i):
        """找到节点i的最小子节点"""
        if i * 2 + 1 > self.size:  # 如果只有左子节点
            return i * 2
        else:
            # 比较左右子节点，返回较小的那个
            if self.heaplist[i * 2] < self.heaplist[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1
    
    def percdown(self, i):
        """下沉操作：将位置i的元素向下调整，维护堆结构"""
        while (i * 2) <= self.size:  # 当i至少有左子节点时
            mc = self.minchild(i)    # 找到最小子节点
            # 如果当前节点大于最小子节点，交换它们
            if self.heaplist[i] > self.heaplist[mc]:
                self.heaplist[i], self.heaplist[mc] = self.heaplist[mc], self.heaplist[i]
            i = mc  # 移动到子节点位置
    
    def delmin(self):
        """删除并返回堆顶的最小元素"""
        m = self.heaplist[1]  # 保存堆顶元素（最小值）
        # 将最后一个元素移到堆顶
        self.heaplist[1] = self.heaplist[self.size]
        self.size -= 1  # 堆大小减少
        self.heaplist.pop()  # 移除最后一个元素
        self.percdown(1)     # 对堆顶元素进行下沉操作
        return m  # 返回最小值
    
    def buildheap(self, alist):
        i = len(alist) // 2  # 从最后一个非叶子节点开始
        self.size = len(alist)
        # 修正：正确构建堆列表（添加0索引占位）
        self.heaplist = [0] + alist[:]  # 创建1-indexed堆
        
        # 从最后一个非叶子节点开始下沉调整
        while i > 0:
            self.percdown(i)
            i -= 1

# 主程序
n = int(input())  # 读取操作次数
h = xiaoheap()    # 创建小顶堆实例

for _ in range(n):
    s = input()   # 读取操作指令
    
    if s[0] == '1':  # 插入操作
        s = s.split()
        a = int(s[1])  # 提取要插入的数字
        h.insert(a)     # 插入堆中
    
    else:  # 删除最小元素操作
        print(h.delmin())  # 删除并打印最小值
```
### 拒绝手搓

Python 提供了内置的 `heapq` 模块来实现堆（优先队列）数据结构。

#### 基本用法

##### 最小堆（默认）

```python
import heapq

# 创建一个空堆（实际上是列表）
heap = []

# 插入元素
heapq.heappush(heap, 5)    # 堆: [5]
heapq.heappush(heap, 2)    # 堆: [2, 5] -> 自动调整
heapq.heappush(heap, 10)   # 堆: [2, 5, 10]
heapq.heappush(heap, 1)    # 堆: [1, 2, 10, 5]

# 弹出最小元素
print(heapq.heappop(heap))  # 输出: 1, 堆变为 [2, 5, 10]
print(heapq.heappop(heap))  # 输出: 2, 堆变为 [5, 10]

# 查看最小元素（不弹出）
print(heap[0])  # 输出: 5
```

##### 最大堆（使用负值技巧）

```python
import heapq

# 创建最大堆（存储元素的负值）
max_heap = []

# 插入元素（存储负值）
heapq.heappush(max_heap, -10)  # 堆: [-10]
heapq.heappush(max_heap, -5)   # 堆: [-10, -5]
heapq.heappush(max_heap, -15)  # 堆: [-15, -5, -10]

# 弹出最大元素（取负值得到原值）
print(-heapq.heappop(max_heap))  # 输出: 15
print(-heapq.heappop(max_heap))  # 输出: 10
```

##### 堆化现有列表

```python
import heapq

# 现有列表
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# 堆化列表（原地转换）
heapq.heapify(nums)
print("堆化后的列表:", nums)  # 输出: [1, 1, 2, 3, 5, 9, 4, 6]

# 弹出所有元素
while nums:
    print(heapq.heappop(nums), end=" ")  # 输出: 1 1 2 3 4 5 6 9
```

##### 同时插入和弹出元素

```python
import heapq

heap = [3, 5, 1, 2, 7]
heapq.heapify(heap)

# heappushpop = 先push再pop
print(heapq.heappushpop(heap, 4))  # 输出: 1（当前堆: [2, 3, 4, 5, 7]）

# heapreplace = 先pop再push
print(heapq.heapreplace(heap, 0))  # 输出: 2（当前堆: [0, 3, 4, 5, 7]）
```

##### 获取最大/最小的 n 个元素

```python
import heapq

scores = [85, 92, 78, 60, 95, 88, 73, 99]

# 获取最高的3个分数
print("最高分:", heapq.nlargest(3, scores))  # 输出: [99, 95, 92]

# 获取最低的2个分数
print("最低分:", heapq.nsmallest(2, scores))  # 输出: [60, 73]
```

##### 处理复杂对象（使用元组）

```python
import heapq

# 任务队列：每个任务有优先级和描述
tasks = []

# 添加任务（优先级，描述）
heapq.heappush(tasks, (3, "低优先级任务"))
heapq.heappush(tasks, (1, "高优先级任务"))
heapq.heappush(tasks, (2, "中优先级任务"))

# 处理任务（按优先级顺序）
while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"处理任务: {task} (优先级: {priority})")
```

输出：
```
处理任务: 高优先级任务 (优先级: 1)
处理任务: 中优先级任务 (优先级: 2)
处理任务: 低优先级任务 (优先级: 3)
```

#### heapq 模块关键函数

| 函数 | 描述 | 时间复杂度 |
|------|------|-----------|
| `heapq.heappush(heap, item)` | 将元素推入堆 | O(log n) |
| `heapq.heappop(heap)` | 弹出最小元素 | O(log n) |
| `heapq.heapify(list)` | 将列表转换为堆（原地） | O(n) |
| `heapq.heappushpop(heap, item)` | 先推入再弹出 | O(log n) |
| `heapq.heapreplace(heap, item)` | 先弹出再推入 | O(log n) |
| `heapq.nlargest(k, iterable)` | 返回最大的 k 个元素 | O(n log k) |
| `heapq.nsmallest(k, iterable)` | 返回最小的 k 个元素 | O(n log k) |


## 并查集

并查集（Disjoint Set Union, **DSU**）是一种高效的数据结构，用于处理**不相交集合的合并与查询**问题。


### **一、核心操作**
1. **MakeSet(x)**：创建一个新集合，包含单个元素 `x`。
2. **Find(x)**：找到元素 `x` 所在集合的代表（根节点）。
3. **Union(x, y)**：合并包含 `x` 和 `y` 的两个集合。

---

### **二、Python 实现**
以下是一个带路径压缩和按秩合并优化的并查集实现：

```python
class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))  # 父节点数组
        self.rank = [0] * size           # 按秩合并：记录树的高度

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return  # 已在同一集合中
        # 按秩合并：将小树合并到大树上
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
```

---

### **三、优化策略**
1. **路径压缩（Path Compression）**：
   - 在 `Find` 操作中，将查询路径上的所有节点直接指向根节点，减少树的高度。
   - **效果**：使后续的 `Find` 操作更快。

2. **按秩合并（Union by Rank/Size）**：
   - 合并时总是将较小的树合并到较大的树上，避免树的高度增加。
   - **效果**：保持树的平衡性，降低查找时间。

---

### **四、应用场景**
#### **1. 判断图的连通性**
- **问题**：给定无向图中的边，判断任意两节点是否连通。
- **示例**：
  ```python
  dsu = DisjointSet(5)
  dsu.union(0, 1)
  dsu.union(2, 3)
  dsu.union(1, 3)
  print(dsu.find(0) == dsu.find(4))  # False（0和4不在同一连通分量）
  ```

#### **2. Kruskal 算法中的环检测**
- **作用**：在构造最小生成树时，检测添加边是否会形成环。
- **逻辑**：如果两个节点的根相同，则添加该边会形成环。

#### **3. 社交网络好友关系**
- **应用**：快速判断两个人是否属于同一社交圈。

#### **4. 像素连通性（图像处理）**
- **应用**：标记图像中的连通区域（如岛屿、背景等）。

---

### **五、经典例题**
#### **例题1：无向图中检测环**
```python
def has_cycle(n, edges):
    dsu = DisjointSet(n)
    for u, v in edges:
        if dsu.find(u) == dsu.find(v):  # 如果已连通，添加边会形成环
            return True
        dsu.union(u, v)
    return False

# 示例输入
edges = [[0, 1], [1, 2], [2, 0]]  # 形成三角形环
print(has_cycle(3, edges))  # 输出: True
```

#### **例题2：朋友圈问题**
```python
def find_circle_num(friends):
    n = len(friends)
    dsu = DisjointSet(n)
    for i in range(n):
        for j in range(i + 1, n):
            if friends[i][j] == 'Y':
                dsu.union(i, j)
    # 统计有多少个不同的根节点
    return len(set(dsu.find(i) for i in range(n)))

# 示例输入
friends = [
    ['N', 'Y', 'N'],
    ['Y', 'N', 'N'],
    ['N', 'N', 'N']
]
print(find_circle_num(friends))  # 输出: 2
```

---

### **七、扩展技巧**
1. **带权并查集**：
   - 维护额外信息（如节点到根的距离），适用于处理带权值的连通性问题。
   - 示例：判断无向图中是否存在负权环。

2. **离散化处理**：
   - 当元素不是连续整数时，需先对元素进行离散化映射。

---


### **例题**

#### **题目要求**
给定一个无向图和多个查询，判断每个查询中的两个节点是否连通。

#### **解题步骤**
1. **初始化并查集**  
   每个节点初始时是自己的父节点，秩为0。
   ```python
   parent = list(range(n))
   rank = [0] * n
   ```

2. **处理所有边**  
   对每条边 $(u, v)$，执行 `union(u, v)`，将两个节点合并到同一集合。
   ```python
   for u, v in edges:
       union(u, v)
   ```

3. **处理所有查询**  
   对每个查询 $(a, b)$，执行 `find(a) == find(b)`，判断两个节点是否属于同一集合。
   ```python
   result = []
   for a, b in queries:
       result.append(find(a) == find(b))
   ```

---

#### **示例解析**
假设输入为：
- `n = 4`（节点数）
- `edges = [[0,1], [2,3]]`
- `queries = [[0,2], [1,3], [0,1]]`

**步骤1：初始化**
- `parent = [0,1,2,3]`
- `rank = [0,0,0,0]`

**步骤2：处理边**
- **合并 0 和 1**：
  - `find(0) = 0`, `find(1) = 1`
  - 两棵树的秩相同（均为0），将 `parent[1] = 0`，并 `rank[0] += 1` → `rank = [1,0,0,0]`
- **合并 2 和 3**：
  - `find(2) = 2`, `find(3) = 3`
  - 两棵树的秩相同（均为0），将 `parent[3] = 2`，并 `rank[2] += 1` → `rank = [1,0,1,0]`

**步骤3：处理查询**
- **查询 [0,2]**：
  - `find(0)`：路径压缩后 `parent[0] = 0`
  - `find(2)`：路径压缩后 `parent[2] = 2`
  - 根不同 → 返回 `False`
- **查询 [1,3]**：
  - `find(1)` → 递归找到根 0
  - `find(3)` → 递归找到根 2
  - 根不同 → 返回 `False`
- **查询 [0,1]**：
  - `find(0)` 和 `find(1)` 均为 0 → 根相同 → 返回 `True`

**最终输出**：
```python
[False, False, True]
```

---

### **并查集的优化技巧**
1. **路径压缩**  
   在 `find` 操作中，将路径上的所有节点直接指向根节点，大幅减少后续查找的时间。

2. **按秩合并**  
   总是将较小的树合并到较大的树上，避免树的高度增加过多，保持树的平衡。

---




---

## Dijkstra算法
### **一、Dijkstra算法的核心思想**
1. **贪心策略**：每次从未确定最短路径的节点中，选择距离起点最近的节点作为“当前节点”，并更新其邻接节点的距离。
2. **动态更新**：通过当前节点的最短路径，动态刷新其邻接节点的最短路径估计值。
3. **优先队列**：使用优先队列（最小堆）高效选择当前最短路径的节点。

---

### **二、算法步骤详解**

#### **输入**
- 一个带权图 $ G = (V, E) $，其中 $ V $ 是节点集合，$ E $ 是边集合。
- 起始节点 $ s $。

#### **初始化**
- `dist[]`：记录从起点 $ s $ 到每个节点的最短距离，初始时设为无穷大（∞），`dist[s] = 0`。
- `visited[]`：标记节点是否已确定最短路径。
- 优先队列（堆）：按距离排序，初始时插入 `(dist[s], s)`。

#### **算法流程**
1. **取出堆顶节点**：从优先队列中取出当前距离起点最近的节点 $ u $。
2. **跳过已处理节点**：如果 $ u $ 已被访问过（`visited[u] = true`），跳过。
3. **标记已处理**：将 $ u $ 标记为已访问。
4. **更新邻接节点**：遍历 $ u $ 的所有邻接节点 $ v $：
   - 如果通过 $ u $ 到 $ v $ 的路径比当前记录的 `dist[v]` 更短，则更新 `dist[v]`。
   - 将更新后的 `dist[v]` 和 $ v $ 插入优先队列。
5. **重复**：直到优先队列为空或所有节点都被处理。

---

### **三、Python实现示例**

#### **邻接表表示法**
```python
import heapq

def dijkstra(graph, start):
    # 初始化距离和访问标记
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    visited = set()
    priority_queue = [(0, start)]  # (距离, 节点)

    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph[u].items():
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                heapq.heappush(priority_queue, (dist[v], v))
    
    return dist
```

#### **示例输入**
```python
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 1, 'D': 1},
    'C': {'A': 5, 'B': 1, 'D': 2},
    'D': {'B': 1, 'C': 2, 'E': 4},
    'E': {'D': 4}
}

start = 'A'
print(dijkstra(graph, start))
```

#### **输出**
```python
{
    'A': 0,
    'B': 2,
    'C': 3,
    'D': 3,
    'E': 7
}
```

---

### **四、算法特性与优化**

#### **时间复杂度**
- **邻接表 + 优先队列**：$ O((V + E) \log V) $，其中 $ V $ 是节点数，$ E $ 是边数。
- **邻接矩阵**：$ O(V^2) $，适合稠密图（边数接近 $ V^2 $）。

#### **优化技巧**
1. **优先队列优化**：使用斐波那契堆（Fibonacci Heap）可将时间复杂度降低到 $ O(E + V \log V) $。
2. **路径记录**：维护一个 `prev[]` 数组，记录最短路径的前驱节点，从而回溯路径。

---

### **五、应用场景**

1. **网络路由**：计算两个节点之间的最短路径（如OSPF协议）。
2. **地图导航**：汽车导航系统（如Google Maps）中的路径规划。
3. **社交网络**：分析用户之间的最短关系链。
4. **资源分配**：在带权图中优化资源分配路径。

---

### **六、注意事项**

1. **负权边限制**：Dijkstra算法**不能处理负权边**，因为贪心策略可能导致错误。若存在负权边，需改用 Bellman-Ford 算法。
2. **图的表示**：邻接表适合稀疏图，邻接矩阵适合稠密图。
3. **无限循环**：若优先队列未正确管理（如重复插入节点），可能导致死循环。

---

### **八、总结**

Dijkstra算法是图论中最基础且实用的算法之一，其核心在于贪心策略与优先队列的结合。通过逐步扩展最短路径，它能够高效地解决单源最短路径问题。掌握其实现与优化技巧，是解决实际路径规划问题的关键。


散列函数（Hash Function）和二次探测法（Quadratic Probing）是数据结构中用于高效存储和检索数据的关键技术。以下是对它们的详细解释：

---

## **散列函数（Hash Function）**

### **1. 定义与作用**
散列函数是一种将**任意长度的输入**（如字符串、整数等）映射为**固定长度输出**（称为哈希值或散列值）的算法。其核心作用是：
- **快速定位数据**：通过哈希函数将关键字转换为数组索引，直接访问存储位置。
- **减少冲突**：设计良好的散列函数能尽可能均匀地分布关键字，降低冲突概率。

### **2. 特性**
1. **确定性**：
   - 相同输入始终生成相同的哈希值。
2. **均匀性**：
   - 不同输入应尽可能生成不同的哈希值，避免聚集。
3. **高效性**：
   - 计算速度快，适合大规模数据处理。
4. **抗碰撞性**：
   - 难以找到两个不同输入生成相同的哈希值（密码学散列函数要求更高）。

### **3. 常见散列函数**
- **除留余数法**：`H(key) = key % m`，其中 `m` 是散列表长度。
- **平方取中法**：取关键字平方后的中间几位作为哈希值。
- **折叠法**：将关键字分割成几部分，叠加后取模。


---

### **二次探测法（Quadratic Probing）**

#### **1. 定义与原理**
二次探测法是**开放寻址法**（Open Addressing）的一种冲突解决策略。当发生哈希冲突时，它通过**平方数序列**逐步寻找下一个可用位置，避免线性探测法的聚集问题。

#### **2. 冲突解决公式**
假设初始哈希地址为 `h0 = H(key)`，第 `i` 次探测的位置为：
$$
h_i = (h_0 + i^2) \mod m \quad \text{或} \quad h_i = (h_0 - i^2) \mod m
$$
其中：
- `i` 是探测次数（从 1 开始）。
- `m` 是散列表长度（通常为质数）。

#### **3. 探测步骤**
1. **计算初始地址**：`h0 = H(key)`。
2. **检查冲突**：
   - 如果 `h0` 为空，直接插入。
   - 如果冲突，按公式依次尝试 `h0 ± 1², h0 ± 2², h0 ± 3², ...`。
3. **插入成功**：找到第一个空闲位置后插入数据。

#### **4. 示例**
假设散列表长度 `m = 11`，关键字 `key = 29`：
- 初始哈希地址：`h0 = 29 % 11 = 7`。
- 若位置 7 被占用，则依次尝试：
  - `h1 = (7 + 1²) % 11 = 8`
  - `h2 = (7 + 2²) % 11 = 10`
  - `h3 = (7 + 3²) % 11 = 3`
  - 直到找到空位置。

#### **5. 优点**
- **减少聚集**：相比线性探测法（连续探测），二次探测法的步长非线性增长，减少了相邻位置的冲突。
- **空间利用率高**：适合散列表长度为质数且负载因子（α = 已用槽位 / 总槽位）不超过 0.5 的场景。

#### **6. 缺点**
- **可能无法覆盖所有位置**：若 `m` 不是质数，某些探测序列可能无法遍历所有槽位。
- **探测次数增加**：在极端情况下，可能需要多次探测才能找到空位。

---

#### **7. 代码实现示例（Python）**
```python
def quadratic_probing_insert(table, key, m):
    h0 = key % m
    i = 1
    while table[(h0 + i*i) % m] is not None:
        i += 1
    table[(h0 + i*i) % m] = key
```
### 例题：用二次探查法建立散列表
#### 描述
给定一系列整型关键字和素数P，用除留余数法定义的散列函数H（key)=key%M，将关键字映射到长度为M的散列表中，用二次探查法解决冲突.

本题不涉及删除，且保证表长不小于关键字总数的2倍，即没有插入失败的可能。

#### 输入
输入第一行首先给出两个正整数N（N<=1000）和M（一般为>=2N的最小素数），分别为待插入的关键字总数以及散列表的长度。
第二行给出N个整型的关键字。数字之间以空格分隔。
#### 输出
在一行内输出每个整型关键字的在散列表中的位置。数字间以空格分隔。
#### 样例输入
5 11
24 13 35 15 14
#### 样例输出
2 3 1 4 7 
#### 提示
探查增量序列依次为：1^2，-1^2，2^2 ，-2^2，....,^2表示平方

注意输入可能是重复的，需要注意

#### 代码：

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]

hashmap=[None]*m
ans=[]
for num in num_list:
    i=0
    H=num%m
    while True:
        if hashmap[(H+i*i)%m] is None or hashmap[(H+i*i)%m]==num:
            hashmap[(H+i*i)%m]=num
            ans.append((H+i*i)%m)
            break
        if hashmap[(H-i*i)%m] is None or hashmap[(H-i*i)%m]==num:
            hashmap[(H-i*i)%m]=num
            ans.append((H-i*i)%m)
            break
        i+=1
print(*ans)
        
        
```

---


## 最小生成树
### **一、核心概念**
- **生成树（Spanning Tree）**  
  一个连通图的生成树是包含图中所有顶点的无环连通子图（即一棵树）。
  
- **最小生成树（MST）**  
  所有生成树中，边的权值之和最小的那棵树。

---

### **二、算法与实现**
#### **1. Kruskal 算法**
- **核心思想**：按边权从小到大选择边，确保不形成环。
- **步骤**：
  1. 将所有边按权值升序排序。
  2. 依次选择最小的边，若该边连接的两个顶点不在同一连通分量中，则加入 MST。
  3. 使用**并查集（Union-Find）**高效判断是否形成环。
- **时间复杂度**：\(O(E \log E)\)（排序边的时间复杂度主导）。
- **适用场景**：稀疏图（边数 \(E\) 接近顶点数 \(V\)）。
- **示例**：
```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False  # 已连通，不需要合并
        # 按秩合并
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
        return True

def kruskal_mst(edges, num_vertices):
    edges.sort(key=lambda x: x[2])  # 按边权升序排序
    uf = UnionFind(num_vertices)
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            if len(mst) == num_vertices - 1:
                break  # 生成树边数达到 V-1 时停止

    return mst, total_weight

# 示例输入（边格式：u, v, weight）
edges = [
    (0, 1, 1),  # A-B
    (0, 2, 4),  # A-C
    (1, 2, 2),  # B-C
    (1, 3, 6),  # B-D
    (2, 3, 3),  # C-D
]
num_vertices = 4  # 顶点数（A=0, B=1, C=2, D=3）

mst_edges, total = kruskal_mst(edges, num_vertices)
print("Kruskal MST 边集合:", mst_edges)  # 输出: [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
print("总权重:", total)  # 输出: 6
```

#### **2. Prim 算法**
- **核心思想**：从任意顶点出发，逐步扩展当前生成树的边集，每次选择权值最小的边。
- **步骤**：
  1. 初始化一个顶点集合，任选一个顶点加入。
  2. 每次选择连接集合内外顶点且权值最小的边，将该顶点加入集合。
  3. 使用**优先队列（堆）**优化选择最小边的过程。
- **时间复杂度**：\(O(E + V \log V)\)（使用斐波那契堆优化）。
- **适用场景**：稠密图（边数 \(E\) 远大于顶点数 \(V\)）。
- **示例**：
```python
import heapq

def prim_mst(graph, num_vertices):
    # 初始化优先队列，从顶点0开始
    heap = []
    visited = [False] * num_vertices
    mst = []
    total_weight = 0

    # 初始顶点0的邻接边加入堆
    visited[0] = True
    for neighbor, weight in graph[0]:
        heapq.heappush(heap, (weight, 0, neighbor))

    while heap and len(mst) < num_vertices - 1:
        weight, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        # 将边(u, v)加入MST
        visited[v] = True
        mst.append((u, v, weight))
        total_weight += weight
        # 将v的邻接边加入堆
        for neighbor, w in graph[v]:
            if not visited[neighbor]:
                heapq.heappush(heap, (w, v, neighbor))

    return mst, total_weight

# 示例输入（邻接表格式）
graph = {
    0: [(1, 1), (2, 4)],  # A的邻接边：A-B(1), A-C(4)
    1: [(0, 1), (2, 2), (3, 6)],  # B的邻接边：B-A(1), B-C(2), B-D(6)
    2: [(0, 4), (1, 2), (3, 3)],  # C的邻接边：C-A(4), C-B(2), C-D(3)
    3: [(1, 6), (2, 3)],  # D的邻接边：D-B(6), D-C(3)
}
num_vertices = 4

mst_edges, total = prim_mst(graph, num_vertices)
print("Prim MST 边集合:", mst_edges)  # 输出: [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
print("总权重:", total)  # 输出: 6
```
---

### **三、关键性质**
1. **切割性质（Cut Property）**  
   在一个图的任意切割（将顶点分为两个不相交集合）中，权值最小的横跨切割的边一定属于 MST。

2. **环性质（Cycle Property）**  
   在一个图中，任意环中权值最大的边一定不属于 MST。

3. **唯一性**  
   - 如果图中所有边的权值**互不相同**，则 MST 唯一。
   - 若存在相同权值的边，可能生成多个不同的 MST。

---

### **四、示例**
假设一个图包含 4 个顶点（A, B, C, D）和以下边：  
- A-B: 1  
- A-C: 4  
- B-C: 2  
- B-D: 6  
- C-D: 3  

**Kruskal 算法步骤**：  
1. 按边权排序：A-B(1), B-C(2), C-D(3), A-C(4), B-D(6)  
2. 依次选择边：A-B → B-C → C-D。总权值 = 1 + 2 + 3 = 6。  

**Prim 算法步骤**（从 A 开始）：  
1. 初始顶点集合 {A}，候选边：A-B(1), A-C(4) → 选 A-B。  
2. 顶点集合 {A, B}，候选边：B-C(2), B-D(6) → 选 B-C。  
3. 顶点集合 {A, B, C}，候选边：C-D(3) → 选 C-D。总权值 = 6。  

---

## **BFS（广度优先搜索）**  模板
>解决迷宫类问题的通用 Python 模板，适用于寻找从起点到终点的**最短路径**或**判断是否可达**：

### 模板1
```python
from typing import List
from collections import deque

def maze_bfs(maze: List[List[int]], start: tuple, end: tuple) -> int:
    """
    迷宫BFS模板
    :param maze: 二维迷宫矩阵，0表示可通行，1表示障碍
    :param start: 起点坐标 (x, y)
    :param end: 终点坐标 (x, y)
    :return: 最短路径步数（若不可达返回-1）
    """
    # 边界检查
    if not maze or not maze[0]:
        return -1
    rows, cols = len(maze), len(maze[0])
    
    # 起点或终点不可达
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return -1
    
    # 方向数组：上下左右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 初始化队列和访问标记
    queue = deque()
    queue.append((start[0], start[1], 0))  # (x, y, steps)
    visited = set()
    visited.add((start[0], start[1]))
    
    while queue:
        x, y, steps = queue.popleft()
        
        # 到达终点
        if (x, y) == end:
            return steps
        
        # 遍历四个方向
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 检查新坐标是否合法且未被访问
            if 0 <= nx < rows and 0 <= ny < cols:
                if maze[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))
    
    # 队列为空仍未到达终点
    return -1

```

### 模板2
```python
from collections import deque
from typing import List, Tuple

class Solution:
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上下左右四个方向

    def min_moves(self, matrix: List[str]) -> int:
        m, n = len(matrix), len(matrix[0])
        
        # 特殊情况：起点或终点是障碍物
        if matrix[0][0] == '#' or matrix[m-1][n-1] == '#':
            return -1

        # 初始化距离矩阵，INF 表示不可达
        INF = float('inf')
        distance = [[INF] * n for _ in range(m)]
        distance[0][0] = 0  # 起点距离为 0

        # 初始化队列，从起点开始
        queue = deque([(0, 0)])

        # BFS 遍历
        while queue:
            x, y = queue.popleft()
            current_dist = distance[x][y]

            # 如果到达终点，返回当前距离
            if x == m - 1 and y == n - 1:
                return current_dist

            # 遍历四个方向
            for dx, dy in self.DIRECTIONS:
                nx, ny = x + dx, y + dy

                # 检查是否在网格范围内且不是障碍物
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#':
                    # 如果找到更短的路径，更新距离并加入队列
                    if current_dist + 1 < distance[nx][ny]:
                        distance[nx][ny] = current_dist + 1
                        queue.append((nx, ny))

        # 如果无法到达终点
        return -1
```


## **Bellman-Ford 算法及其变体**

### **一、标准 Bellman-Ford 算法**

#### **1. 核心思想**
- **单源最短路径**，允许图中存在 **负权边**。
- **V-1 次松弛操作**：更新所有可能的最短路径。
- **负权环检测**：第 V 次松弛仍可更新，则存在负权环。

#### **2. 适用场景**
- 图中存在负权边（但无负权环）。
- 需要检测负权环。

#### **3. 模板代码**

```python
from collections import defaultdict
import sys

def bellman_ford(n, edges, src):
    INF = float('inf')
    distance = [INF] * n
    distance[src] = 0

    # 松弛 V-1 次
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if distance[u] != INF and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                updated = True
        if not updated:
            break  # 提前退出优化

    # 检测负权环
    has_negative_cycle = False
    for u, v, w in edges:
        if distance[u] != INF and distance[u] + w < distance[v]:
            has_negative_cycle = True
            break

    return distance, has_negative_cycle
```

#### **4. 输入输出说明**
- **输入**：
  - `n`: 顶点数。
  - `edges`: 边列表，格式为 `[(u, v, w), ...]`。
  - `src`: 起点。
- **输出**：
  - `distance`: 源点到各点的最短距离。
  - `has_negative_cycle`: 是否存在负权环。

---

### **二、SPFA（队列优化的 Bellman-Ford）**

#### **1. 核心思想**
- 使用 **队列** 优化松弛过程，仅对“被更新的顶点”进行处理。
- 平均时间复杂度接近 `O(E)`，最坏仍为 `O(V*E)`。

#### **2. 适用场景**
- 图中存在大量负权边，但无负权环。
- 动态图（边权重频繁变化）。

#### **3. 模板代码**

```python
from collections import deque, defaultdict
import sys

def spfa(n, edges, src):
    INF = float('inf')
    distance = [INF] * n
    distance[src] = 0
    in_queue = [False] * n
    queue = deque([src])
    in_queue[src] = True
    cnt = [0] * n  # 统计入队次数，用于检测负权环

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v, w in edges[u]:
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    cnt[v] += 1
                    if cnt[v] >= n:  # 检测负权环
                        return None  # 存在负权环

    return distance
```

#### **4. 输入输出说明**
- **输入**：
  - `edges`: 邻接表形式，`edges[u] = [(v, w), ...]`。
- **输出**：
  - `distance`: 最短路径数组，若返回 `None` 表示存在负权环。

---

### **三、限制步数的 Bellman-Ford 变体（LeetCode 787 问题）**

#### **1. 核心思想**
- 在标准 Bellman-Ford 基础上，**限制最多 k+1 次飞行**（即最多 k 次中转）。
- 使用二维数组 `dist[i][v]` 表示经过 `i` 次飞行后到达 `v` 的最小成本。

#### **2. 适用场景**
- 限制最多 `k` 次中转（如航班问题）。
- 无法使用 Dijkstra（因为有负权边）。

#### **3. 模板代码**

```python
from typing import List
from collections import defaultdict
import sys

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, w in flights:
            graph[u].append((v, w))

        INF = float('inf')
        # dist[i][v]: 经过 i 次飞行后到达 v 的最小费用
        dist = [[INF] * n for _ in range(k + 2)]
        dist[0][src] = 0  # 初始没有飞行时，只有起点可达

        for i in range(1, k + 2):
            for u in range(n):
                for v, w in graph[u]:
                    if dist[i-1][u] != INF and dist[i-1][u] + w < dist[i][v]:
                        dist[i][v] = dist[i-1][u] + w

        min_cost = min(dist[i][dst] for i in range(k + 2))
        return min_cost if min_cost != INF else -1
```

#### **4. 输入输出说明**
- **输入**：
  - `n`: 城市数。
  - `flights`: 航班列表，格式为 `[[u, v, w], ...]`。
  - `src`: 起点城市。
  - `dst`: 目的城市。
  - `k`: 最多中转次数。
- **输出**：
  - 最低票价，若无法到达返回 `-1`。

---


### **四、常见错误与注意事项**

1. **初始化问题**：
   - 起点的距离必须初始化为 0。
   - 所有其他点初始化为 `INF`。
2. **负权环检测**：
   - 标准 Bellman-Ford 需额外一轮松弛操作。
   - SPFA 通过入队次数判断。
3. **限制步数的变体**：
   - 外层循环应限制为 `k+1` 次。
   - 使用二维数组记录不同步数的最短距离。

---


## 最大流问题

最大流问题是网络优化中的一个经典问题，其目标是在一个有向加权图中找到从源点（source）到汇点（sink）的最大流量。图中的每条边都有一个容量限制，表示该边最多可以通过的流量。

#### 定义

- **图**：\( G = (V, E) \)，其中 \( V \) 是节点集合，\( E \) 是边集合。
- **容量函数**：\( c(u, v) \)，表示从节点 \( u \) 到节点 \( v \) 的边的最大容量。
- **流量函数**：\( f(u, v) \)，表示从节点 \( u \) 到节点 \( v \) 的实际流量。
- **源点**：\( s \in V \)，起始节点。
- **汇点**：\( t \in V \)，结束节点。

#### 目标

最大化从源点 \( s \) 到汇点 \( t \) 的总流量，同时满足以下条件：
1. 流量守恒：对于每个中间节点 \( v \neq s, t \)，流入 \( v \) 的总流量等于流出 \( v \) 的总流量。
2. 容量限制：对于每条边 \( (u, v) \in E \)，流量 \( f(u, v) \leq c(u, v) \)。

### Ford-Fulkerson 方法

Ford-Fulkerson 方法是一种用于计算最大流的经典算法。它通过不断寻找增广路径并增加流量来逐步逼近最大流。具体步骤如下：

1. **初始化**：将所有边的流量初始化为 0。
2. **查找增广路径**：使用 BFS 或 DFS 查找从源点 \( s \) 到汇点 \( t \) 的增广路径。增广路径是一条从 \( s \) 到 \( t \) 的路径，且路径上的每条边的剩余容量大于 0。
3. **计算路径流量**：在找到增广路径后，计算路径上的最小剩余容量。
4. **更新流量**：沿着增广路径增加相应流量，并更新反向边的流量（即减少反向边的流量以允许未来的流量回溯）。
5. **重复**：重复上述过程直到找不到增广路径为止。

#### 具体实现

以下是使用 Ford-Fulkerson 方法的 Python 实现：
```python
from collections import deque

def bfs(capacity, source, sink, parent):
    visited = [False] * len(capacity)
    queue = deque([source])
    visited[source] = True
    
    while queue:
        u = queue.popleft()
        
        for v in range(len(capacity)):
            if not visited[v] and capacity[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

def ford_fulkerson(capacity, source, sink):
    parent = [-1] * len(capacity)
    max_flow = 0
    
    while bfs(capacity, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        
        # 找到路径上的最小剩余容量
        while s != source:
            path_flow = min(path_flow, capacity[parent[s]][s])
            s = parent[s]
        
        # 更新路径上每条边的剩余容量，并更新反向边的流量
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = parent[v]
        
        # 将路径流量加到最大流中
        max_flow += path_flow
    
    return max_flow

# 示例用法
if __name__ == "__main__":
    # 定义图的容量矩阵
    # 图表示为一个二维列表，其中 capacity[i][j] 是从节点 i 到节点 j 的容量
    capacity = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ]
    
    source = 0  # 源节点
    sink = 5    # 汇节点
    
    result = ford_fulkerson(capacity, source, sink)
    print("最大可能流:", result)

```


### 代码解释

1. **bfs 函数**：
   - 使用队列进行广度优先搜索，查找从源点 \( s \) 到汇点 \( t \) 的增广路径。
   - 如果找到增广路径，则返回 `True` 并记录路径中的父节点信息。

2. **ford_fulkerson 函数**：
   - 初始化父节点数组 `parent` 和最大流 `max_flow`。
   - 在每次循环中调用 `bfs` 函数查找增广路径。
   - 如果找到增广路径，计算路径上的最小剩余容量 `path_flow`。
   - 更新路径上每条边的剩余容量，并更新反向边的流量。
   - 将 `path_flow` 加到 `max_flow` 中。
   - 重复上述过程直到找不到增广路径为止。

3. **Example usage**：
   - 定义一个示例图的容量矩阵 `capacity`。
   - 设置源点 `source` 和汇点 `sink`。
   - 调用 `ford_fulkerson` 函数计算最大流并打印结果。




