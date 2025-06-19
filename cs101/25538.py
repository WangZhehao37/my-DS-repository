n=int(input())
a=bin(n)[2:]
judge=1
for i in range(len(a)//2):
    if a[i] != a[-i-1]:
        judge=0
        break
if judge:
    print("Yes")
else :
    print("No")