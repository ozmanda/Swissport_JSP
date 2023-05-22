from pandas import to_datetime, Timedelta


def minutes_since_midnight(time):
    midnight = to_datetime('00:00', format='%H:%M')
    return int((time - midnight) / Timedelta(minutes=1))


def minutes_to_midnight(time):
    midnight = to_datetime('00:00', format='%H:%M') + Timedelta(days=1)
    return int((midnight - time) / Timedelta(minutes=1))


def calculate_maximum_delay(aircraft):
    max_delay = []
    for ac in aircraft:
        max_delay.append(minutes_to_midnight(aircraft[ac]['STD']))
    return max_delay
