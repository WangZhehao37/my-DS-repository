while True:
    N,M=map(int,input().split())
    if N==M==0:
        break
    count_={}
    maxcount=1
    for i in range(N):
        s=[int(j) for j in input().split()]
        for num in s:
            if num in count_:
                count_[num]+=1
                if count_[num]>maxcount:
                    maxcount=count_[num]
            else :
                count_[num]=1
    ans=[]
    sec=1
    for key,val in count_.items():
        if val<maxcount and val >sec:
            ans=[key]
            sec=val
        elif val==sec:
            ans.append(key)
    print(*sorted(ans))