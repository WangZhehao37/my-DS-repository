n=int(input())
if n%2==0:
    sum_=[0 for i in range(n//2)]
    for j in range(0,n//2):
        list=input().split()
        for i in range(n//2):
            if i<j:
                sum_[i]+=int(list[i])+int(list[n-i-1])
            else :
                sum_[j]+=int(list[i])+int(list[n-i-1])
    for j in range(n//2,n):
        list=input().split()
        for i in range(n//2):
            if i<n-j-1:
                sum_[i]+=int(list[i])+int(list[n-i-1])
            else :
                sum_[n-j-1]+=int(list[i])+int(list[n-i-1])
    print(max(sum_))
if n%2==1:
    sum_=[0 for i in range(n//2+1)]
    for j in range(0,n//2):
        list=input().split()
        for i in range(n//2):
            if i<j:
                sum_[i]+=int(list[i])+int(list[n-i-1])
            else :
                sum_[j]+=int(list[i])+int(list[n-i-1])
        sum_[j]+=int(list[n//2])
    for j in range(n//2,n):
        list=input().split()
        for i in range(n//2):
            if i<n-j-1:
                sum_[i]+=int(list[i])+int(list[n-i-1])
            else :
                sum_[n-j-1]+=int(list[i])+int(list[n-i-1])
        sum_[n-j-1]+=int(list[n//2])
    print(max(sum_))
