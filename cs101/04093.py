N = int(input())
count_ = {}
for i in range(N):
    s = input().split()
    for j in range(len(s)-1):
        sj = int(s[j+1])
        if sj in count_:
            if i+1 not in count_[sj]:
                count_[sj].append(i+1)
        else:
            count_[sj] = [i+1]
M = int(input())
for i in range(M):
    ans = []
    judge = [int(_) for _ in input().split()]
    for X, Y in count_.items():
        judge_X = True
        for j in range(N):
            if judge[j] == 1:
                if j+1 not in Y:
                    judge_X = False
                    break
            elif judge[j] == -1:
                if j+1 in Y:
                    judge_X = False
                    break
        if judge_X == True:
            ans.append(X)
    if ans:
        print(*sorted(ans))
    else:
        print("NOT FOUND")
