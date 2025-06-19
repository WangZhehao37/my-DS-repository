a=[int(i) for i in input().split()]
min_price=float('inf')
profit=0
for i in a :
    if i<min_price:
        min_price=i
    elif i-min_price>profit:
        profit=i-min_price
if profit<=0:
    print(0)
else :
    print(profit)
