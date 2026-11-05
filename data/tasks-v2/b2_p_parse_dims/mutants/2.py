def parse_dims(text):
 if not isinstance(text,str) or not text: raise ValueError()
 parts=text.split('x')
 if any(not p or not p.isascii() or not p.isdigit() or p[0]=='0' for p in parts): raise ValueError()
 return [int(p) for p in parts]
