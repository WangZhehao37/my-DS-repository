
m=int(input())
n=0
countmap=[]
# (-correct,total,name,correctmap)
hashdic={}
for i in range(m):
    a,b,c=input().split(',')
    if a not in hashdic:
        n+=1
        hashdic[a]=n-1
        countmap.append([0,0,a,[]])
    if c=="no":
        countmap[hashdic[a]][1]+=1
    elif c=="yes":
        countmap[hashdic[a]][1]+=1
        if b not in countmap[hashdic[a]][3]:
            countmap[hashdic[a]][0]-=1
            countmap[hashdic[a]][3].append(b)
            
countmap.sort()
for i in range(12):
    if i<len(countmap):
        print(f'{i+1} {countmap[i][2]} {-countmap[i][0]} {countmap[i][1]}')
    
