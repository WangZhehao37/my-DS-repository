a=input()
b=input()
c=''
d=0
if len(a)>len(b):
    b='0'*(len(a)-len(b))+b
elif len(b)>len(a):
    a='0'*(len(b)-len(a))+a
for i in range(len(a)-1,-1,-1):
    c=str((int(a[i])+int(b[i])+d)%10)+c
    d=(int(a[i])+int(b[i])+d)//10
c=str(d)+c
for i in range(len(c)):
    if c[i] != '0':
        break
for j in c[i:]:
    print(j,end='')