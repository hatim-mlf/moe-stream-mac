import re
def parse_env_lines(text):
    result={}
    pattern=re.compile(r"[A-Z_][A-Z0-9_]*")
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("missing equals sign")
        key,value=line.split("=",1)
        if pattern.fullmatch(key) is None:
            raise ValueError("invalid key")
        if key not in result:
            result[key]=value
    return result
