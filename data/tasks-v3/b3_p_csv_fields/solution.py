import csv
import io

def parse_csv_record(text):
    try:
        rows=list(csv.reader(io.StringIO(text,newline=""),strict=True))
    except csv.Error as error:
        raise ValueError("malformed CSV") from error
    if len(rows)!=1: raise ValueError("expected one record")
    return rows[0]
