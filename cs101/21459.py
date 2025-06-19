def cal(a):
    if a % 2 == 1 :
        print(f"{a}*3+1={3*a+1}")
        return cal(3*a+1)
    else :
        print(f"{a}/2={a//2}")
        if a==2:
            return 0
        else  :
            return cal(a//2)
cal(int(input()))