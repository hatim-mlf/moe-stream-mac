def parse_size_bytes(text):
 units={"B":1,"KiB":1024,"MiB":1024**2,"GiB":1024**3}
 if not isinstance(text,str): raise ValueError()
 u=next((u for u in ("GiB","MiB","KiB","B") if text.endswith(u)),None)
 if u is None: raise ValueError()
 d=text[:-len(u)]
 if not d or not d.isascii() or not d.isdigit(): raise ValueError()
 return int(d)*units[u]
