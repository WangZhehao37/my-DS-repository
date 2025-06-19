move = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),(1, -2), (1, 2), (2, -1), (2, 1)]
paths = []


def judgemove(x, y, board):
    return 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == -1


def backtrack(m, n, x, y, pos, board, path):
    if pos == m*n:
        paths.append(path)
        return True
    for dy, dx in move:
        nx = x+dx
        ny = y+dy
        if judgemove(nx, ny, board):
            board[nx][ny] = pos
            path += f'{chr(ny+65)}{nx+1}'
            if backtrack(m, n, nx, ny, pos+1, board, path):
                return True
            board[nx][ny] = -1
            path=path[:-2]
    return False


def solve_tour(n, m, s_x, s_y):
    board = [[-1 for _ in range(m)] for __ in range(n)]
    board[s_x][s_y] = 0
    if not backtrack(m, n, s_x, s_y, 1, board, f'{chr(s_y+65)}{s_x+1}'):
        return 0
    else:
        sorted(paths)
        return paths[0]


n = int(input())
for i in range(n):
    paths=[]
    p, q = map(int, input().split())
    flag = False
    print(f"Scenario #{i+1}:")
    for y in range(q):
        if flag:
            break
        for x in range(p):
            ans = solve_tour(p, q, x, y)       
            if ans != 0:
                print(ans)
                flag=True
                break
    if flag==False:
        print('impossible')
    print()