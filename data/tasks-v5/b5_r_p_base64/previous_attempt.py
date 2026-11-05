import base64

def decode_base64_ascii(text):
    if not isinstance(text,str): raise ValueError
    return base64.b64decode(text)
