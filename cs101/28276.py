class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0]*size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            
dsu=DisjointSet(26)
n=int(input())
judgelist=[]
flag=True
for i in range(n):
    s=input()
    if s[1:3]=='==':
        dsu.union(ord(s[0])-ord('a'),ord(s[3])-ord('a'))
    else:
        judgelist.append((ord(s[0])-ord('a'),ord(s[3])-ord('a')))
for a,b in judgelist:
    if  dsu.find(a)==dsu.find(b):
        print('False')
        flag=False
        break
if flag:
    print('True')
