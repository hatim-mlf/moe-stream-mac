def count_nonblank_lines(text):
    if not isinstance(text,str): raise ValueError
    return sum(bool(line.strip()) for line in text.splitlines())
