def escape_tsv_cell(text):
 return text.replace("\\","\\\\").replace("\t","\\t").replace("\n","\\n")
