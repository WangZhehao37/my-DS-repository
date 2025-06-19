def count_path(n, m, x, y):
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    visited = [[False] * m for _ in range(n)]
    def dfs(x, y, step):
        if step == n * m:
            return 1
        count = 0
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
                visited[nx][ny] = True
                count += dfs(nx, ny, step + 1)
                visited[nx][ny] = False  
        
        return count
    
    visited[x][y] = True
    result = dfs(x, y, 1)
    visited[x][y] = False 
    return result

T = int(input())
for _ in range(T):
    n, m, x_0, y_0 = map(int, input().split())
    print(count_path(n, m, x_0, y_0))
