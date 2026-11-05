def format_duration(total_seconds):
    h,r=divmod(total_seconds,3600)
    m,s=divmod(r,60)
    return "{}:{}:{}".format(h,m,s)
