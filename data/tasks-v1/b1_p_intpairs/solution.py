import re
def parse_int_pairs(text):
    result=[]
    for line in text.splitlines():
        if not line.strip():
            continue
        pieces=line.split(":")
        if len(pieces)!=2:
            raise ValueError("expected one colon")
        fields=[piece.strip() for piece in pieces]
        if any(re.fullmatch(r"[+-]?[0-9]+",field) is None for field in fields):
            raise ValueError("invalid integer field")
        result.append((int(fields[0],10),int(fields[1],10)))
    return result
