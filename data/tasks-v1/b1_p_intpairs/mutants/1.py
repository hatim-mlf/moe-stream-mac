def parse_int_pairs(text):
    result=[]
    for line in text.splitlines():
        line=line.strip()
        if line:
            left,right=line.split(":")
            result.append((int(left),int(right)))
    return result
