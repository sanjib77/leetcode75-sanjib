def largestAltitude(gain):
    ans = cur = 0
    for g in gain:
        cur += g
        ans = max(ans, cur)
    return ans
