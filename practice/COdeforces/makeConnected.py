def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        grid.append(input().strip())
    
    # Find all black cells
    black_cells = set()
    for i in range(n):
        for j in range(n):
            if grid[i][j] == '#':
                black_cells.add((i, j))
    
    # Check for existing 3 consecutive in initial grid
    def has_three_consecutive():
        for i in range(n):
            for j in range(n):
                # Check horizontal
                if j + 2 < n:
                    if grid[i][j] == '#' and grid[i][j+1] == '#' and grid[i][j+2] == '#':
                        return True
                # Check vertical
                if i + 2 < n:
                    if grid[i][j] == '#' and grid[i+1][j] == '#' and grid[i+2][j] == '#':
                        return True
        return False
    
    if has_three_consecutive():
        return "NO"
    
    if len(black_cells) == 0:
        return "YES"
    
    # Check if painting (i,j) would create 3 consecutive with ORIGINAL black cells
    def can_paint(i, j):
        # Check all patterns where (i,j) would be the middle or edge of 3 consecutive
        patterns = [
            # Horizontal patterns: need 2 other original black cells in line
            [(i, j-2), (i, j-1)],  # ##X pattern
            [(i, j-1), (i, j+1)],  # #X# pattern  
            [(i, j+1), (i, j+2)],  # X## pattern
            # Vertical patterns
            [(i-2, j), (i-1, j)],  # two above
            [(i-1, j), (i+1, j)],  # one above, one below
            [(i+1, j), (i+2, j)]   # two below
        ]
        
        for pattern in patterns:
            if all(0 <= pi < n and 0 <= pj < n and (pi, pj) in black_cells for pi, pj in pattern):
                return False
        
        return True
    
    from collections import deque
    
    # BFS to check connectivity
    start = next(iter(black_cells))
    visited = {start}
    queue = deque([start])
    
    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in visited:
                # Can visit if it's already black OR we can safely paint it
                if (ni, nj) in black_cells or can_paint(ni, nj):
                    visited.add((ni, nj))
                    queue.append((ni, nj))
    
    return "YES" if black_cells.issubset(visited) else "NO"

t = int(input())
for _ in range(t):
    print(solve())