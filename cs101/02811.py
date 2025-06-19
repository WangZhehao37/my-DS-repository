light=[]
move=[(1,0),(-1,0),(0,1),(0,-1),(0,0)]
for i in range(5):
    light.append([int(i) for i in input().split()])
ans=[[0]*6 for _ in range(5)]
for i in range(5):
    for j in range(6):
        for dx,dy in move:
            x=i+dx
            y=j+dy
            if 0<=x<5 and 0<=y<6:
                ans[i][j]+=light[x][y]
        ans[i][j]= ans[i][j]%2
for i in ans:
    print(*i)