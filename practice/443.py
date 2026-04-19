def compress(chars):
    i = 0
    pos = 0
    while i < len(chars):
        j = i
        while j < len(chars) and chars[j] == chars[i]:
            j += 1
        chars[pos] = chars[i]
        pos += 1
        if j - i > 1:
            for c in str(j - i):
                chars[pos] = c
                pos += 1
        i = j
    return pos
