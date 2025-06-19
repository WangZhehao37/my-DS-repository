n=int(input())
cubes=[i**3 for i in range(0,n+1)]
if n >= 6 :
    for a in range(6,n+1):
        for b in range(2,a):
            if cubes[b]*3>cubes[a]:
                break
            for c in range(b,a):
                if cubes[b]+cubes[c]*2>cubes[a]:
                    break
                for d in range(c,a):
                    if cubes[b]+cubes[c]+cubes[d]==cubes[a]:
                        print(f"Cube = {a}, Triple = ({b},{c},{d})")
                        break
                    elif cubes[b]+cubes[c]+cubes[d]>cubes[a]:
                        break