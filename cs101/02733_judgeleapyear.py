from calendar import isleap
n=int(input())
"""if n % 3200 ==0 :
    print("N")
elif n % 100 ==0 and n % 400 !=0 :
    print("N")
elif n % 4 ==0 :
    print("Y")
else:
    print("N")"""
if isleap(n):
    print("Y")
else :
    print("N")