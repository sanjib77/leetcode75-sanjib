def solve():
    n, k, x = map(int, input().split())
    friends = list(map(int, input().split()))
    unique_friends = sorted(set(friends))
    
    def can_achieve(min_dist):
        merged = []
        for a in unique_friends:
            lo = max(0, a - min_dist + 1)
            hi = min(x, a + min_dist - 1)
            if lo <= hi:
                if merged and lo <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
        
        forbidden_count = sum(hi - lo + 1 for lo, hi in merged)
        return (x + 1) - forbidden_count >= k
    
    left, right = 0, x + 1
    best_dist = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_achieve(mid):
            best_dist = mid
            left = mid + 1
        else:
            right = mid - 1
    
    merged = []
    for a in unique_friends:
        lo = max(0, a - best_dist + 1)
        hi = min(x, a + best_dist - 1)
        if lo <= hi:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
    
    teleports = []
    pos = 0
    
    for lo, hi in merged:
        count = min(k - len(teleports), lo - pos)
        if count > 0:
            teleports.extend(range(pos, pos + count))
        pos = hi + 1
        if len(teleports) >= k:
            break
    
    if len(teleports) < k:
        remaining = k - len(teleports)
        teleports.extend(range(pos, pos + remaining))
    
    print(' '.join(map(str, teleports)))

t = int(input())
for _ in range(t):
    solve()