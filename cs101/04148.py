case=1
while True:
    p,e,i,d=map(int,input().split())
    if p==e==i==d==-1:
        break
    for a in range(0,925):
        day=a*23+p
        if (day-e)%28==0 and (day-i)%33==0 and day>d:
            print(f"Case {case}: the next triple peak occurs in {day-d} days.")
            break
    case+=1
        