from pandas import to_datetime, Timedelta

def minutes_since_midnight(time):
    midnight = to_datetime('00:00', format='%H:%M')
    return int((time - midnight) / Timedelta(minutes=1))