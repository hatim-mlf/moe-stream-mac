import base64
import binascii

def decode_base64_ascii(text):
    if not isinstance(text,str): raise ValueError
    try:
        raw=text.encode('ascii')
        decoded=base64.b64decode(raw,validate=True)
    except (UnicodeEncodeError,binascii.Error,ValueError) as exc:
        raise ValueError from exc
    if base64.b64encode(decoded)!=raw: raise ValueError
    return decoded
