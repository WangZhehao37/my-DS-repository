n=int(input())
num=[int(_) for _ in input().split()]
count_tot=0
count_da=0
count_xiao=0
def develop(path,i,n):
    global count_tot
    if 2*i+1>n-1:
        print(*path)
        count_tot+=1
        judge(path)
        return
    if 2*i+2<=n-1:
        path.append(num[2*i+2])
        develop(path,2*i+2,n)
        path.pop()
    if 2*i+1<=n-1:
        path.append(num[2*i+1])
        develop(path,2*i+1,n)
        path.pop()
def judge(path):
    global count_da,count_xiao
    if path[0]>path[1]:
        for i in range(len(path)-1):
            if path[i]<path[i+1]:
                return
        count_da+=1
    if path[0]<path[1]:
        for i in range(len(path)-1):
            if path[i]>path[i+1]:
                return
        count_xiao+=1


develop([num[0]],0,n) 

if count_da==count_tot:
    print("Max Heap")
elif count_xiao==count_tot:
    print("Min Heap")
else :
    print("Not Heap")