def evenly_spaced_indices(length,count):
 if count==0:return []
 return [round(i*(length-1)/(count-1)) for i in range(count)]
