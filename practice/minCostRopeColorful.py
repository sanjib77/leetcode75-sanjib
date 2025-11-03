class Solution:
    def minCost(self, colors: str, neededTime: list[int]) -> int:
        total_cost = 0
        
        for i in range(1, len(colors)):
            if colors[i-1] == colors[i]:
                # Remove the cheaper balloon, keep the expensive one
                total_cost += min(neededTime[i-1], neededTime[i])
                # Update neededTime[i] to the max for next comparison
                neededTime[i] = max(neededTime[i-1], neededTime[i])
        
        return total_cost
                    
a = Solution().minCost("cddcdcae",[4,8,8,4,4,5,4,2] )
print(a)





    
        