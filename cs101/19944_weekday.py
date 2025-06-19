import math
n=int(input())
a=[input() for i in range(n)]
judge={0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"}
for i in range(n):
    c=int(a[i][0:2])
    y=int(a[i][2:4])
    m=int(a[i][4:6])
    d=int(a[i][6:8])
    if m == 1 or m == 2 :
        if y == 0 :
            y=99
            c-=1
            m+=12
        else:
            m+=12
            y-=1
    print(judge[(y+math.floor(y/4)+math.floor(c/4)-2*c+math.floor(2.6*(m+1))+d-1) % 7])