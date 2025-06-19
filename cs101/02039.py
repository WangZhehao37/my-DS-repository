n = int(input())
s = input()
m = len(s)//n
ans = ''
for i in range(n):
    for j in range(m):
        if j % 2 == 0:
            ans = ans+s[i+n*j]
        else:
            ans = ans+s[-i+n*(j+1)-1]
print(ans)
