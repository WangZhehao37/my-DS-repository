pigs=[]
min_pigs=[]
min_=float('inf')
while True:
    try:
        s=input()
        if "push " in s:
            a=int(s[5:])
            pigs.append(a)
            if a<=min_:
                min_=a
                min_pigs.append(a)
        elif s=='pop':
            if pigs:
                b=pigs.pop()
                if b==min_pigs[-1]:
                    min_pigs.pop()
        elif s=='min' and pigs and min_pigs:
            print(min_pigs[-1])
    except EOFError:
        break
