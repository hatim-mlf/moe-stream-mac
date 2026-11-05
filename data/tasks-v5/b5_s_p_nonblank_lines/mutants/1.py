def count_nonblank_lines(text):
    return len([line for line in text.splitlines() if line])
