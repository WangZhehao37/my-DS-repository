import collections

def main():
    n = int(input())
    grid = []
    for _ in range(n):
        row = list(int(x) for x in input())
        grid.append(row)
    
    visited = [[False for _ in range(n)] for _ in range(n)]
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    sr, sc = -1, -1
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                sr, sc = r, c
                break
        if sr != -1:
            break
    
    q = collections.deque()
    q.append((sr, sc))
    visited[sr][sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1 and not visited[nr][nc]:
                    visited[nr][nc] = True
                    q.append((nr, nc))

    for r in range(n):
        for c in range(n):
            if visited[r][c] and grid[r][c] == 1:
                q.append((r, c))
    
    step = 0
    while q:
        curLen = len(q)
        for _ in range(curLen):
            r, c = q.popleft()
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                    visited[nr][nc] = True
                    if grid[nr][nc] == 1:
                        return step
                    q.append((nr, nc))
        step += 1
    return step

print(main())



