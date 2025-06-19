n=int(input())
plus=[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
key={i:str(12-i) for i in range(3,11)}
key.update({0:'1',1:'0',2:"X"})
for i in range(n):
    s=input()
    sum_=0
    for i in range(17):
        sum_+=plus[i]*int(s[i])
    if key[sum_%11]==s[17]:
        print("YES")
    else :
        print("NO")
    