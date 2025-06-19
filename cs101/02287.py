while True:
    n=int(input())
    if n==0:
        break
    tj=[int(i) for i in input().split()]
    tj.sort(reverse=True)
    gw=[int(i) for i in input().split()]
    gw.sort(reverse=True)
    tjup=gwup=0
    tjdown=gwdown=n-1
    i=0
    count=0
    while i<n:
        if tj[tjup]>gw[gwup]:
            i+=1
            count+=1
            tjup+=1
            gwup+=1
        else:
            if tj[tjdown]>gw[gwdown]:
                count+=1
                i+=1
                tjdown-=1
                gwdown-=1
            elif tj[tjdown]==gw[gwup]:
                count-=0
                i+=1
                tjdown-=1
                gwup+=1
            else:
                count-=1
                i+=1
                tjdown-=1
                gwup+=1
    print(200*count)
            
