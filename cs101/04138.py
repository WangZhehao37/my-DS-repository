def euler(n):
    is_prime=[True]*(n+1)
    prime=[]
    
    for i in range(2,n+1):
        if is_prime[i] :
            prime.append(i)
        for p in prime:
            if i * p > n :
                break
            is_prime[i*p]=False
            if i % p == 0:
                break
    return is_prime

s=int(input())
pri=euler(s)
if s == 4 :
    print(4)
elif s%2==0:
    for i in range(3,s//2+1,2):
        if pri[i] and pri[s-i] :
            ans=i*(s-i)
    print(ans)
elif s%2==1:
    print(2*(s-2))