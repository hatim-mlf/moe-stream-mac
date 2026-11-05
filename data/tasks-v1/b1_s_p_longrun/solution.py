def longest_equal_run(values):
    best=current=0
    have_previous=False
    for value in values:
        if have_previous and value==previous:
            current+=1
        else:
            current=1
            previous=value
            have_previous=True
        if current>best:
            best=current
    return best
