def level(a):
    if a in ('+', '-'):
        return 1
    if a in ('*', '/'):
        return 2
    return 0


def judgelevel(a, b):
    return level(a) >= level(b)


def pop_ope(opes, vals):
    ope = opes.pop()
    right = vals.pop()
    left = vals.pop()
    vals.append(f'{left} {right} {ope}')


def trans(s):
    opes = []
    vals = []
    i = 0
    while i < len(s):
        x = s[i]
        if x.isdigit() or x == '.':
            num = x
            while i<len(s)-1 and (s[i+1].isdigit() or s[i+1] == '.'):
                i += 1
                num += s[i]
            vals.append(num)
        elif x == '(':
            opes.append(x)
        elif x == ')':
            while opes[-1] != '(':
                pop_ope(opes, vals)
            opes.pop()
        else:
            while opes and judgelevel(opes[-1], x) and opes[-1] != '(':
                pop_ope(opes, vals)
            opes.append(x)
        i += 1

    while opes:
        pop_ope(opes, vals)

    return vals[0]


n = int(input())
for i in range(n):
    s = input()
    print(trans(s))
