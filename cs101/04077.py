n=int(input())
stack=[]
ans=[]
count=0
def deal(stack,ans,n,m):
    global count
    if len(ans)==n:
        count+=1
        return 0
    if len(stack)!=0:
        a=stack.pop()
        ans.append(a)
        deal(stack,ans,n,m)
        ans.pop()
        stack.append(a)
    if m>0:
        stack.append(n-m+1)
        deal(stack,ans,n,m-1)
        stack.pop()
        
deal(stack,ans,n,n)
print(count)





