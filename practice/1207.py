def uniqueOccurrences(arr):
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return len(freq.values()) == len(set(freq.values()))
