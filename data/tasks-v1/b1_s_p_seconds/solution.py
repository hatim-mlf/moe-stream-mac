def format_duration(total_seconds):
    if isinstance(total_seconds,bool) or not isinstance(total_seconds,int) or total_seconds<0:
        raise ValueError("seconds must be a nonnegative integer")
    hours,rest=divmod(total_seconds,3600)
    minutes,seconds=divmod(rest,60)
    return "{}:{:02d}:{:02d}".format(hours,minutes,seconds)
