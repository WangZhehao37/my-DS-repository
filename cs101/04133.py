d=int(input())
n=int(input())
A=[[0 for i in range(1025)] for j in range(1025)]
for n_i in range(n):
    x,y,i=map(int,input().split())
    for a in range(max(0,x-d),min(1025,x+d+1)):
        for b in range(max(0,y-d),min(1025,y+d+1)):
            A[a][b]+=i
max=-1
count=0
for A_1 in A:
    for A_2 in A_1:
        if A_2>max:
            max=A_2
            count=1
        elif A_2==max:
            count+=1
print(count,max)
        