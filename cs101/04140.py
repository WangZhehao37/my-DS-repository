def f(x):
    return x**3-5*x**2+10*x-80
def f1(x):
    return 3*x**2-10*x+10
x1=6
while True:
    x1,x2=x1-f(x1)/f1(x1),x1
    if -0.00000000001<x1-x2<0.00000000001:
        print(f'{x1:.9f}')
        break