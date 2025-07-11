def tran(lis):
    num_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
        'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000
    }
    ans = 0
    current=0
    for i in lis:
        num = num_map[i]
        if num == 100:
            current = current*num
        elif num>=1000:
            ans+=current*num
            current=0
        else:
            current+=num
    ans+=current
    return ans

s = input()
word = s.split()
if word[0] == "negative":
    print(-tran(word[1:]))
else:
    print(tran(word))
