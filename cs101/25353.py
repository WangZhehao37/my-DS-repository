"""N,D=map(int,input().split())
h=[]
for i in range(N):
    h.append(int(input()))
"""

def min_lexicographical_order(N, D, heights):
    # 初始化分组
    groups = []
    current_group = []

    for i in range(N):
        if current_group and heights[i] - current_group[-1] <= D:
            current_group.append(heights[i])
        else:
            if current_group:
                groups.append(current_group)
            current_group = [heights[i]]
    
    # 添加最后一组
    if current_group:
        groups.append(current_group)
    
    # 对每个组内的身高进行排序
    sorted_groups = [sorted(group) for group in groups]
    
    # 合并所有组内的排序结果
    result = []
    for group in sorted_groups:
        result.extend(group)
    
    return result

# 读取输入
N, D = map(int, input().split())
heights = [int(input()) for _ in range(N)]

# 计算并输出结果
result = min_lexicographical_order(N, D, heights)
for height in result:
    print(height)