from collections import deque

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def bfs(grid, start, end, R, C, K):
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add((start[0], start[1], 0 % K))
    
    while queue:
        x, y, t = queue.popleft()
        if (x, y) == end:
            return t
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < R and 0 <= ny < C:
                next_t = t + 1
                if grid[nx][ny] == "." or (next_t % K == 0 and grid[nx][ny] == "#"):
                    if (nx, ny, next_t % K) not in visited:
                        visited.add((nx, ny, next_t % K))
                        queue.append((nx, ny, next_t))
    
    return "Oop!"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    
    results = []
    for _ in range(T):
        R = int(data[index])
        C = int(data[index + 1])
        K = int(data[index + 2])
        index += 3
        
        grid = []
        start = None
        end = None
        for i in range(R):
            row = list(data[index])
            for j, cell in enumerate(row):
                if cell == "S":
                    start = (i, j)
                elif cell == "E":
                    end = (i, j)
            grid.append(row)
            index += 1
        
        result = bfs(grid, start, end, R, C, K)
        results.append(str(result))
    
    print("\n".join(results))

if __name__ == "__main__":
    main()