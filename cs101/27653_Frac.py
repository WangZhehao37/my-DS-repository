def gcd(m, n):
    while m % n != 0:
        m, n = n, m % n
    return n
class Fraction:
    def __init__(self,top,bottom):
        self.num=top
        self.den=bottom
    def show(self):
        print(f"{self.num}/{self.den}")
    def __str__(self):
        return f"{self.num}/{self.den}"
    def __add__(self,other):
        n_num=self.num*other.den+self.den*other.num
        n_den=self.den*other.den
        b=gcd(n_num,n_den)
        return Fraction(n_num//b,n_den//b)

x,y,z,h=map(int,input().split())
print(Fraction(x,y)+Fraction(z,h))