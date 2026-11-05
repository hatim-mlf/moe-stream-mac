import re
def parse_env_lines(text):
    result={}
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        key,value=line.split("=",1)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*",key): raise ValueError
        result[key]=value
    return result
