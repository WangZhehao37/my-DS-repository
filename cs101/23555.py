n,m1,m2=map(int,input().split())
x=[]
y=[]
z=[[0 for i in range(n)] for _ in range(n)] 
for i in range(m1):
    x.append(tuple(map(int,input().split())))
for i in range(m2):
    y.append(tuple(map(int,input().split())))
for i in x:
    for j in y:
        if i[1]==j[0]:
            z[i[0]][j[1]]+=i[2]*j[2]
for i in range(n):
    for j in range(n):
        if z[i][j]!=0:
            print(f"{i} {j} {z[i][j]}")