n=int(input())
A=[] #老年
B=[] #非老年
for i in range(n):
    id,y=input().split()
    if int(y) >= 60 :
        A.append((id,y,i))
    else :
        B.append((id,y,i))
A.sort(key=lambda x:-int(x[1]))
for i in A:
    print(i[0])
for i in B:
    print(i[0])
    