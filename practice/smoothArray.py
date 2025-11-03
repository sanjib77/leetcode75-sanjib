import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    c = list(map(int, input().split()))
    
    if n == 1:
        print(0)
        return
    
    total = sum(c)
    
    vals = sorted(set(a))
    val_to_idx = {v: i for i, v in enumerate(vals)}
    m = len(vals)
    
    seg = [0] * (4 * m)
    
    def update(node, l, r, pos, val):
        if l == r:
            seg[node] = max(seg[node], val)
            return
        mid = (l + r) // 2
        if pos <= mid:
            update(2*node, l, mid, pos, val)
        else:
            update(2*node+1, mid+1, r, pos, val)
        seg[node] = max(seg[2*node], seg[2*node+1])
    
    def query(node, l, r, ql, qr):
        if qr < l or ql > r or ql > qr:
            return 0
        if ql <= l and r <= qr:
            return seg[node]
        mid = (l + r) // 2
        return max(query(2*node, l, mid, ql, qr), query(2*node+1, mid+1, r, ql, qr))
    
    max_dp = 0
    for i in range(n):
        idx = val_to_idx[a[i]]
        prev_max = query(1, 0, m-1, 0, idx) if m > 0 else 0
        curr_dp = prev_max + c[i]
        update(1, 0, m-1, idx, curr_dp)
        max_dp = max(max_dp, curr_dp)
    
    print(total - max_dp)

t = int(input())
for _ in range(t):
    solve()