def solve():
    n, X = map(int, input().split())
    prices = list(map(int, input().split()))
    
    total = sum(prices)
    k = total // X  # Number of boundaries we can cross
    
    # If no crossings possible
    if k == 0:
        print(0)
        print(' '.join(map(str, prices)))
        return
    
    # Sort prices in descending order
    sorted_prices = sorted(prices, reverse=True)
    
    # Maximum bonus = sum of k largest items
    bonus = sum(sorted_prices[:k])
    
    # Separate into crossing items (k largest) and fillers (rest)
    crossing = sorted_prices[:k]
    fillers = sorted_prices[k:]
    fillers.sort()  # Sort fillers in ascending order
    
    result = []
    S = 0  # Current total spent
    filler_idx = 0
    
    # For each boundary to cross
    for i in range(k):
        target = (i + 1) * X  # Next boundary at target
        
        # Add fillers until S + crossing[i] >= target
        # This ensures the crossing item will actually cross the boundary
        while filler_idx < len(fillers) and S + crossing[i] < target:
            result.append(fillers[filler_idx])
            S += fillers[filler_idx]
            filler_idx += 1
        
        # Add the crossing item
        result.append(crossing[i])
        S += crossing[i]
    
    # Add any remaining fillers
    while filler_idx < len(fillers):
        result.append(fillers[filler_idx])
        filler_idx += 1
    
    print(bonus)
    print(' '.join(map(str, result)))

t = int(input())
for _ in range(t):
    solve()