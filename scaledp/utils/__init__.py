import statistics


def cluster(items, maxgap, key=lambda x: x):
    """Arrange data into groups where successive elements
    differ by no more than *maxgap*
    """
    if not items:
        return []
    items.sort(key=key)
    groups = [[items[0]]]
    for x in items[1:]:
        if abs(key(x) - key(groups[-1][-1])) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def get_size(items, key=lambda x: x):
    if not items:
        return 0
    items = [key(x) for x in items]
    if len(items) == 1:
        return items[0]
    items.sort()
    if len(items) <= 4:
        return int(statistics.mean(items))
    else:
        l = int(len(items) / 4)
        return int(statistics.mode(items[l:-l]))
