import time

def print_timestamp():
    # prints timestamp (day, month, hour, minute)
    print(get_timestamp())

def print_timestamp_seconds():
    # prints timestamp (day, month, hour, minute, second)
    print(get_timestamp_seconds())

def get_timestamp_seconds():
    # returns timestamp as string (day, month, hour, minute, second)
    return time.strftime('%d-%m_%H:%M:%S')

def get_timestamp():
    # returns timestamp as string (day, month, hour, minute)
    return time.strftime('%d-%m_%H:%M')