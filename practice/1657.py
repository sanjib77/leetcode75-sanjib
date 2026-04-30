def closeStrings(word1, word2):
    if len(word1) != len(word2):
        return False
    f1, f2 = {}, {}
    for c in word1:
        f1[c] = f1.get(c, 0) + 1
    for c in word2:
        f2[c] = f2.get(c, 0) + 1
    return set(f1.keys()) == set(f2.keys()) and sorted(f1.values()) == sorted(f2.values())
