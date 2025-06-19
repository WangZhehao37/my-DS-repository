n=int(input())
print(n)
A=('pop', 'no', 'zip', 'zotz', 'tzec', 'xul', 'yoxkin', 'mol', 'chen', 'yax', 'zac', 'ceh', 'mac', 'kankin', 'muan', 'pax', 'koyab', 'cumhu') 
B=('imix', 'ik', 'akbal', 'kan', 'chicchan', 'cimi', 'manik', 'lamat', 'muluk', 'ok', 'chuen', 'eb', 'ben', 'ix', 'mem', 'cib', 'caban', 'eznab', 'canac', 'ahau')
while True:
    try:
        a=input().split(". ")
        day=int(a[0])
        mon,year=a[1].split()
        year=int(year)
        if mon=='uayet':
            tot=year*365+18*20+day
        else :
            tot=year*365+A.index(mon)*20+day
        print(f'{(tot)%13+1} {B[tot%20]} {tot//260}')
    except EOFError:
        break