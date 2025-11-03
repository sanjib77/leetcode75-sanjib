import sys
sys.setrecursionlimit(300000)

def solve():
    n, k = map(int, input().split())
    

    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    

    subtree_size = [0] * (n + 1)
    parent = [0] * (n + 1)
    
    def dfs(node, par):
        parent[node] = par
        subtree_size[node] = 1
        for neighbor in adj[node]:
            if neighbor != par:
                dfs(neighbor, node)
                subtree_size[node] += subtree_size[neighbor]
    
    dfs(1, 0)
    

    total = 0
    threshold = n - k
    
    for v in range(1, n + 1):

        contribution = 1 
        
        for u in adj[v]:

            if u == parent[v]:
                component_size = n - subtree_size[v]
            else:
                component_size = subtree_size[u]
            

            if component_size <= threshold:
                contribution += component_size
        
        total += contribution
    
    print(total)


t = int(input())
for _ in range(t):
    solve()