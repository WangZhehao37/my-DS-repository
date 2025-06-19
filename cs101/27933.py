n=int(input())
stack=[]
i=1
count=0
for j in range(2*n):
    s=input()
    if s[0]=='a':
        num=int(s[4:])
        stack.append(num)
    else:
        while True:
            if stack[-1]==i:
                stack.pop()
                i+=1
                break
            else :
                stack.sort(reverse=True)
                count+=1
print(count)
