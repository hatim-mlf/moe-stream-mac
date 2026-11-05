def parse_size_bytes(text):
    if not isinstance(text, str):
        raise ValueError("expected text")
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    unit = next((u for u in ("GiB", "MiB", "KiB", "B") if text.endswith(u)), None)
    if unit is None:
        raise ValueError("unknown unit")
    digits = text[:-len(unit)]
    if not digits or not digits.isascii() or not digits.isdigit() or (len(digits) > 1 and digits[0] == "0"):
        raise ValueError("invalid count")
    return int(digits) * units[unit]
