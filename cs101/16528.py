n=int(input())
A=[]
for i in range(n):
    b,e=map(int,input().split())
    A.append((e,-b))
A.sort()
last=-1
count=0
for i in range(n):
    if -A[i][1]>last:
        count+=1
        last=A[i][0]
print(count)