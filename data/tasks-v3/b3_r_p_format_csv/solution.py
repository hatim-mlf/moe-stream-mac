import csv
import io

def format_csv_record(fields):
    stream=io.StringIO(newline="")
    csv.writer(stream).writerow(fields)
    return stream.getvalue()[:-2]
