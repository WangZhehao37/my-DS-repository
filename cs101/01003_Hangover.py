def number(x):
    j=2
    while x>0:
        x-=1/j
        j+=1
    return j-2
list=[]
list.append(float(input()))
while list[-1]:
    list.append(float(input()))
for i in list :
    if i :
        print(f"{number(i)} card(s)")
