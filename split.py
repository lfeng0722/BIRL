

def func(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]