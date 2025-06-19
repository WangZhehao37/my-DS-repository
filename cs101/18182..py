ncase = int(input())
for case in range(ncase):
    n, m, b = map(int, input().split())
    A = []
    for i in range(n):
        t, x = map(int, input().split())
        A.append((t, -x))
    A.sort()
    flag=1
    t = -1
    for attack in A:
        if attack[0] > t:
            t = attack[0]
            countm = 1
            b += attack[1]
            if b<=0:
                flag=0
                print(t)
                break
        else:
            if countm==m:
                continue
            else:
                b += attack[1]
                countm+=1
                if b<=0:
                    flag=0
                    print(t)
                    break 
    if flag==1:
        print("alive")