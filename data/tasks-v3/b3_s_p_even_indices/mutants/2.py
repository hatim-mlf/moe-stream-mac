def evenly_spaced_indices(length,count):
 if length<0 or count<0 or count>length: raise ValueError()
 if count==0:return []
 return [(i*length)//count for i in range(count)]
