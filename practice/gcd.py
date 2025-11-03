import math
import sys
input = sys.stdin.readline


CANDIDATES = list(range(2, 51))

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))

    g = a[0]
    for i in range(1, n):
        g = math.gcd(g, a[i])
        if g == 1:  
            break
    ans = -1
    for x in CANDIDATES:
        if math.gcd(g, x) == 1:
            ans = x
            break

    print(ans)