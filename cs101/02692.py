n=int(input())
sum_=0
for _ in range(n):
    judge_all=0
    M={'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'J':0,'L':0,'K':0}
    s=[]
    for i in range(3):
        s.append(input().split())
    for i in range(12):
        if judge_all==1:
            break
        
        M[chr(ord('A')+i)]+=1
        count_=0
        for j in range(3):
            a=0
            b=0
            for aa in s[j][0]:
                a+=M[aa]
            for bb in s[j][1]:
                b+=M[bb]
            if a==b and s[j][2]=="even":
                count_+=1
            if a>b and s[j][2]=="up":
                count_+=1
            if a<b and s[j][2]=="down":
                count_+=1
        if count_==3:
            judge_all=1
            print(f"{chr(ord('A')+i)} is the counterfeit coin and it is heavy.")
            
        M[chr(ord('A')+i)]-=2
        count_=0
        for j in range(3):
            a=0
            b=0
            for aa in s[j][0]:
                a+=M[aa]
            for bb in s[j][1]:
                b+=M[bb]
            if a==b and s[j][2]=="even":
                count_+=1
            if a>b and s[j][2]=="up":
                count_+=1
            if a<b and s[j][2]=="down":
                count_+=1
        if count_==3:
            judge_all=1
            print(f"{chr(ord('A')+i)} is the counterfeit coin and it is light.")
            
        M[chr(ord('A')+i)]+=1
        

        
        