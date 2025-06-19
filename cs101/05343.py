from collections import deque
n=int(input())
poke=deque([i for i in input().split()])
queuenum=[[] for i in range(9)]
queuewor=[[] for i in range(4)]
while poke:
    a=poke.popleft()
    queuenum[int(a[1])-1].append(a)
for i in range(9):
    print(f'Queue{i+1}:',end='')
    if queuenum[i]:
        poke.extend(queuenum[i])
        print(*queuenum[i])
    else:
        print('')
while poke:
    a=poke.popleft()
    queuewor[ord(a[0])-ord('A')].append(a)
for i in range(4):
    print(f"Queue{chr(ord('A')+i)}:",end='')
    if queuewor[i]:
        poke.extend(queuewor[i])
        print(*queuewor[i])
    else:
        print('')
print(*poke)