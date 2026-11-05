import base64
import binascii

def decode_base64_ascii(text):
    if not isinstance(text,str): raise ValueError
    try: return base64.b64decode(text.encode('ascii'),validate=True)
    except (UnicodeEncodeError,binascii.Error) as exc: raise ValueError from exc
