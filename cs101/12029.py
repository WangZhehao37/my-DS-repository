from typing import List, Tuple

DIRECTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))

def dfs_iterative(x: int, y: int, height: int, grid: List[List[int]], water_height: List[List[int]]) -> None:
    rows, cols = len(grid), len(grid[0])
    stack = [(x, y)]
    water_height[x][y] = height

    while stack:
        cx, cy = stack.pop()

        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] < height and water_height[nx][ny] < height:
                water_height[nx][ny] = height
                stack.append((nx, ny))

def can_flood(start_x: int, start_y: int, points: List[Tuple[int, int]], grid: List[List[int]]) -> str:
    rows, cols = len(grid), len(grid[0])
    water_height = [[0] * cols for _ in range(rows)]

    for px, py in points:
        if grid[px][py] > grid[start_x][start_y]:
            dfs_iterative(px, py, grid[px][py], grid, water_height)

    return "Yes" if water_height[start_x][start_y] > 0 else "No"

def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    idx = 0
    num_cases = data[idx]
    idx += 1
    results = []

    for _ in range(num_cases):
        rows, cols = data[idx:idx + 2]
        idx += 2

        grid = [data[idx + i * cols:idx + (i + 1) * cols] for i in range(rows)]
        idx += rows * cols

        start_x, start_y = data[idx] - 1, data[idx + 1] - 1
        idx += 2

        num_points = data[idx]
        idx += 1

        points = [(data[idx + i * 2] - 1, data[idx + i * 2 + 1] - 1) for i in range(num_points)]
        idx += num_points * 2

        results.append(can_flood(start_x, start_y, points, grid))

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    main()