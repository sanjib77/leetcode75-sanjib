def predictPartyVictory(senate):
    rad = []
    dir = []
    for i, c in enumerate(senate):
        if c == 'R':
            rad.append(i)
        else:
            dir.append(i)
    n = len(senate)
    while rad and dir:
        if rad[0] < dir[0]:
            rad.append(rad[0] + n)
        else:
            dir.append(dir[0] + n)
        rad.pop(0)
        dir.pop(0)
    return "Radiant" if rad else "Dire"
