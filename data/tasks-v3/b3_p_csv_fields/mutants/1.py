import csv,io
def parse_csv_record(text): return next(csv.reader(io.StringIO(text)))
