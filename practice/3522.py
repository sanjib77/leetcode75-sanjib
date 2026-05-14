def calculateScore(instructions, values):
    n = len(instructions)
    seen = [False] * n
    ans = 0
    i = 0
    while 0 <= i < n and not seen[i]:
        seen[i] = True
        if instructions[i] == "add":
            ans += values[i]
            i += 1
        else:
            i += values[i]
    return ans
