import math
a=[]
while True:
    a.append([int(i) for i in input().split()])
    if sum(a[-1])==0:
        break
for i in range(0,len(a)-1):
    b=a[i]
    sum_=0
    sum_+=b[5]
    sum_+=b[4]
    b[0]=b[0]-b[4]*11
    sum_+=b[3]
    if b[1] > b[3]*5 :
        b[1]=b[1]-b[3]*5
    else :
        b[0]=b[0]-(b[3]*20-b[1]*4)
        b[1]=0
    sum_+=math.ceil(b[2]/4)
    if b[2] % 4 == 1:
        if b[1] > 5 :
            b[1]=b[1]-5
            b[0]=b[0]-7
        else :
            b[0]=b[0]-(27-b[1]*4)
            b[1]=0
    elif b[2] % 4 == 2:
        if b[1] > 3 :
            b[1]=b[1]-3
            b[0]=b[0]-6
        else :
            b[0]=b[0]-(18-b[1]*4)
            b[1]=0
    elif b[2] % 4 == 3:
        if b[1] > 1 :
            b[1]=b[1]-1
            b[0]=b[0]-5
        else :
            b[0]=b[0]-(9-b[1]*4)
            b[1]=0
    if b[1] > 0 :
        sum_+=math.ceil(b[1]/9)
        b[0]=b[0]-(36-(b[1]%9)*4)
    if b[0]>0:
        sum_+=math.ceil(b[0]/36)
    print(sum_)