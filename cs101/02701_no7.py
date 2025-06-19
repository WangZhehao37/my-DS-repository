n=int(input())
sum_=0
for i in range(1,n+1):
    if '7' in str(i):
        continue
    elif i % 7 ==0 :
        continue
    else :
        sum_+=i**2
print(sum_)