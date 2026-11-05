def escape_tsv_cell(text):
 return text.replace("\t","\\t").replace("\r","\\r").replace("\n","\\n").replace("\\","\\\\")
