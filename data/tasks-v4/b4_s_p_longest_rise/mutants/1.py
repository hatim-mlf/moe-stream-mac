def longest_strict_rise(values):
    best=current=0; previous=None; seen=False
    for value in values:
        if seen and value>=previous: current+=1
        else: current=1
        best=max(best,current); previous=value; seen=True
    return best
