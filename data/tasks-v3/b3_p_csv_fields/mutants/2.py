import csv,io
def parse_csv_record(text):
 rows=list(csv.reader(io.StringIO(text,newline=""),strict=True))
 if len(rows)!=1: raise ValueError()
 return rows[0]
