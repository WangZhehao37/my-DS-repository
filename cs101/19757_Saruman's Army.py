while True:
    R,n=map(int,input().split())
    if R==n==-1:
        break
    else :
        army=sorted([int(i) for i in input().split()])
        i=0
        min_=1001
        ans=0
        max_=-1
        while i<n:
            if army[i]>max_ and army[i]<min_:
                min_=army[i]
            
            if min_<army[i]-R:
                ans+=1
                max_=army[i-1]+R
                min_=1001
            else :
                i+=1
    if army[-1]<=max_:
        print(ans)
    else :
        print(ans+1)
                