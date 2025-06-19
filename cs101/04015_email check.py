while True:
    try:
        code=input()
        judge=0
        if code.count("@")!=1:
            print("NO")
        else :
            if code[0]=="@" or code[0]=="." or code[-1]=="@" or code[-1]=="." :
                print("NO")
            else :
                b=code.index("@")
                if code[b-1]=="." or code[b+1]=="." :
                    print("NO")
                else :
                    if code[b+2:].count(".")>=1:
                        print("YES")
                    else :
                        print("NO")     
    except EOFError:
        break