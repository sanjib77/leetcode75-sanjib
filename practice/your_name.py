q= int(input())

for i in range(q):
    n = int(input())
    s , t = input().split()
    if sorted(s) == sorted(t):
        print("YES")
    else:
        print("NO")