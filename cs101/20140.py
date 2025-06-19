s = input()
stack = []
num = 0
string = ''
for i in s:
    if i.isdigit():
        num = num*10+int(i)
    elif i == '[':
        stack.append((num, string))
        num = 0
        string = ''
    elif i == ']':
        num_, string_ = stack.pop()
        string = string_ + string * num
        num = num_
    else:
        string += i
print(string)
