from collections import deque

def bfs_maze(maze, start, end):
    """
    使用 BFS 找到迷宫中最短路径的步数。

    参数:
    - maze: 二维列表，. 表示可通过，# 表示障碍
    - start: 起点坐标 (row, col)
    - end: 终点坐标 (row, col)

    返回:
    - 如果可以到达终点，返回最短步数；否则返回 -1
    """
    rows = len(maze)
    cols = len(maze[0])

    # 四个方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 初始化队列，保存 (行号, 列号)
    queue = deque()
    queue.append((start[0], start[1]))

    # 记录已访问的格子
    visited = [[False] * cols for _ in range(rows)]
    visited[start[0]][start[1]] = True

    # 记录到达每个点的步数
    distance = [[-1] * cols for _ in range(rows)]
    distance[start[0]][start[1]] = 0

    while queue:
        x, y = queue.popleft()

        # 如果到达终点，返回步数
        if (x, y) == end:
            return distance[x][y]

        # 遍历四个方向
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # 检查是否合法：边界、是否为路、是否未访问
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != "#" and not visited[nx][ny]:
                visited[nx][ny] = True
                distance[nx][ny] = distance[x][y] + 1
                queue.append((nx, ny))

    # 如果无法到达终点
    return -1

def find_startandend(maze):
    rows = len(maze)
    cols = len(maze[0])
    for i in range(rows):
        for j in range(cols) :
            if maze[i][j]=="S":
                start=(i,j)
            if maze[i][j]=="E":
                end=(i,j)
    return start,end

def main():
    T=int(input())
    for i in range(T):
        R,C=map(int,input().split())
        maze=[]
        for j in range(R):
            line=input()
            maze.append(list(line.strip()))
        start,end=find_startandend(maze)
        ans=bfs_maze(maze, start, end)
        if ans==-1:
            print("oop!")
        else :
            print(ans)
            

if __name__ == "__main__":
    main()  