t = int(input())
for i in range(t):
    a, b, c, d = map(int, input().split())
    if (a == b) and (c == d) and (a == d):
        print("Yes")
    else:
        print("No")