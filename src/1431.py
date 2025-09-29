class Solution:
    def kidsWithCandies(self, candies: list[int], extraCandies: int) -> list[bool]:
        maxVal = max(candies)
        # return [c + extraCandies >= maxVal for c in candies] 
        result = []
        for c in range(len(candies)):
            if candies[c] + extraCandies >= maxVal:
                result.append(True)
            else:
                result.append(False)
        return result
    
print(Solution().kidsWithCandies([2,3,5,1,3], 3))  # [True,True,True,False,True]
print(Solution().kidsWithCandies([4,2,1,1,2], 1))  # [True,False,False,False,False]
print(Solution().kidsWithCandies([12,1,12], 10)) # [True,False,True]