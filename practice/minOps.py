def solve():
    n = int(input())
    a = list(map(int, input().split()))
    unique_count = len(set(a))
    print(2 * unique_count - 1)

t = int(input())
for i in range(t):
    solve()