k=int(input())
s=input()
for i in s:
    if 'a'<=i<='z':
        print(chr((ord(i)-k-ord('a'))%26+ord('a')),end='')
    if 'A'<=i<='Z':
        print(chr((ord(i)-k-ord('A'))%26+ord('A')),end='')