import csv,io
def format_csv_record(fields):
 s=io.StringIO(newline="");csv.writer(s).writerow(fields);return s.getvalue()
