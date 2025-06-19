N,K=map(int,input().split())
list=[]
for i in range(N):
    a,b=map(int,input().split())
    list.append((a,b,i))
list.sort(reverse=True)
anslist=sorted(list[0:K],key=lambda x :x[1],reverse=True)
print(anslist[0][2]+1)