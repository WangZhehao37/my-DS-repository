import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]

hashmap=[None]*m
ans=[]
for num in num_list:
    i=0
    H=num%m
    while True:
        if hashmap[(H+i*i)%m] is None or hashmap[(H+i*i)%m]==num:
            hashmap[(H+i*i)%m]=num
            ans.append((H+i*i)%m)
            break
        if hashmap[(H-i*i)%m] is None or hashmap[(H-i*i)%m]==num:
            hashmap[(H-i*i)%m]=num
            ans.append((H-i*i)%m)
            break
        i+=1
print(*ans)
        
        