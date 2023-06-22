import time

def time_func(func):
    '''
        Show the runtime for the function
    '''
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        runtime = time.time() - start
        print(f"{func.__name__}(): runtime = {runtime}")
        return result
    return wrapper


def log_func(func):
    '''
        Print the function name and arguments passed
    '''
    def wrapper(*args, **kwargs):
        print (f"{func.__name__}({args})")


def memorize(func):
    '''
        Save the output from the function so that subsequent calls will
        access the pre-computer output
    '''
    # TODO: Check for locally saved pickel file for function calls
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            result = func(*args)
            cache[args] = result
            return result
        
        
def retry(max_tries=5, delay=15, max_runtime=320):
    '''
        Retries a function call for a given number of retries, or max execution time
    '''
    def retry_func(func):
        def retry(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                tries = 0
                start = time.time()

                while tries < max_tries and int(time.time() - start) < max_runtime:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        tries += 1
                        if tries == max_tries:
                            print (f"Error: {func.__name__} Max reties"
                            raise e
                        time.sleep(delay)
                return wrapper
            return retry
            
    
    
