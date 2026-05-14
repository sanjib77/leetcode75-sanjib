def isGood(nums):
    cnt = {}
    for x in nums:
        cnt[x] = cnt.get(x, 0) + 1
    n = len(nums) - 1
    if cnt.get(n, 0) != 2:
        return False
    for i in range(1, n):
        if cnt.get(i, 0) != 1:
            return False
    return True
