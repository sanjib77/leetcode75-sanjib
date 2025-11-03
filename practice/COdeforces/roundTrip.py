def solve():
    R0, X, D, n = map(int, input().split())
    rounds = input().strip()
    
    min_r = R0  # Minimum possible rating
    max_r = R0  # Maximum possible rating
    count = 0
    
    for round_type in rounds:
        if round_type == '1':
            # Div.1: always rated for everyone
            count += 1
            # After rated round, range expands
            min_r = max(0, min_r - D)  # Can decrease by up to D (but not negative)
            max_r = max_r + D          # Can increase by up to D
        else:
            # Div.2: rated only if rating < X
            if min_r < X:
                # We can participate by choosing a rating < X
                count += 1
                # Constrain our rating to be < X, then expand from there
                new_min = max(0, min_r - D)
                new_max = min(max_r, X - 1) + D
                min_r = new_min
                max_r = new_max
            # If min_r >= X, we can't participate, rating unchanged
    
    return count

t = int(input())
for _ in range(t):
    print(solve())