def transpose_rows(rows):
    iterator=iter(rows)
    try:
        first=list(next(iterator))
    except StopIteration:
        return []
    columns=[[value] for value in first]
    for row in iterator:
        values=list(row)
        if len(values)!=len(columns):
            raise ValueError("ragged rows")
        for column,value in zip(columns,values):
            column.append(value)
    return columns
