def cal(x):
    if x==1:
        print("End")
    elif x%2==0 :
        print(f'{x}/2={x//2}')
        cal(x//2)
    elif x%2==1 :
        print(f'{x}*3+1={3*x+1}')
        cal(3*x+1)

n=int(input())
cal(n)
