def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Find first and last occurrence of each value, and count
    first = {}
    last = {}
    count = {}
    
    for i, val in enumerate(a):
        if val not in first:
            first[val] = i
            count[val] = 0
        last[val] = i
        count[val] += 1
    
    # Get all distinct values sorted
    values = sorted(count.keys())
    k = len(values)
    
    if k <= 1:
        print(0)
        return
    
    # DP: dp[i] = max elements we can keep using values[0..i]
    dp = [0] * k
    dp[0] = count[values[0]]
    
    for i in range(1, k):
        v = values[i]
        prev_v = values[i-1]
        
        # Option 1: Don't take current value
        dp[i] = dp[i-1]
        
        # Option 2: Take current value
        take_v = count[v]
        
        if prev_v == v - 1:
            # Consecutive values - check position conflict
            if last[v] < first[prev_v]:
                # No conflict (all v before all prev_v)
                take_v += dp[i-1]
            else:
                # Conflict - skip previous value
                if i >= 2:
                    take_v += dp[i-2]
        else:
            # Not consecutive - no conflict
            take_v += dp[i-1]
        
        dp[i] = max(dp[i], take_v)
    
    print(n - dp[k-1])

t = int(input())
for _ in range(t):
    solve()