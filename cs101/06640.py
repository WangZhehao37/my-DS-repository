N = int(input())
count_ = {}
for i in range(N):
    s = input().split()
    for j in range(len(s)-1):
        if s[j+1] in count_:
            if i+1 not in count_[s[j+1]]:
                count_[s[j+1]].append(i+1)
        else:
            count_[s[j+1]] = [i+1]
M = int(input())
for i in range(M):
    find = input()
    if find not in count_:
        print("NOT FOUND")
    else:
        print(*sorted(count_[find]))
