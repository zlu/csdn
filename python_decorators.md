### Python Decorators: A Comprehensive Guide

A Python decorator is a function that modifies the behavior of another function (or class/method) without changing its source code. Decorators are often used for logging, timing, access control, memoization, etc.  In this tutorial, we will learn about decorators, how they work, and how to create and use them effectively in different applications.

#### Basic Decorator Syntax
```python
def timer_decorator(func):
    from time import time
    
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Function {func.__name__} took {end - start:.2f} seconds")
        return result
    
    return wrapper

@timer_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done!"

slow_function()
```
And the result is:
```python
>>> slow_function()
Function slow_function took 1.00 seconds
'Done!'
```

#### How Decorators Actually Work

The `@timer_decorator` syntax is actually syntactic sugar. Here's what happens step by step:

**Step 1: Decorator Definition**
```python
def timer_decorator(func):
    # This function takes another function as an argument
    # and returns a new function (wrapper)
    
    def wrapper(*args, **kwargs):
        # This is the new function that will replace the original
        start = time()
        result = func(*args, **kwargs)  # Call the original function
        end = time()
        print(f"Function {func.__name__} took {end - start:.2f} seconds")
        return result
    
    return wrapper  # Return the wrapper function
```

**Step 2: What @timer_decorator Does**
The `@timer_decorator` syntax is equivalent to:
```python
def slow_function():
    import time
    time.sleep(1)
    return "Done!"

# This line does the same thing as @timer_decorator
slow_function = timer_decorator(slow_function)
```

**Step 3: The Transformation Process**
```python
# Before decoration
def slow_function():
    import time
    time.sleep(1)
    return "Done!"

# After decoration (what actually happens)
def slow_function():
    # This is now the wrapper function
    start = time()
    result = original_slow_function()  # Calls the original function
    end = time()
    print(f"Function slow_function took {end - start:.2f} seconds")
    return result
```

**Step 4: Function Call Flow**
```python
# When you call slow_function(), this happens:
slow_function()
# ↓
wrapper()  # The wrapper function is called
# ↓
start = time()
result = original_slow_function()  # Original function executes
end = time()
print(f"Function slow_function took {end - start:.2f} seconds")
return result
```

#### Visual Representation of Decorator Execution

```python
# 1. Define the decorator
def timer_decorator(func):
    print(f"Decorator called with function: {func.__name__}")
    
    def wrapper(*args, **kwargs):
        print(f"Wrapper called for {func.__name__}")
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Function {func.__name__} took {end - start:.2f} seconds")
        return result
    
    print(f"Returning wrapper for {func.__name__}")
    return wrapper

# 2. Apply the decorator
@timer_decorator
def slow_function():
    print("Original function executing...")
    time.sleep(1)
    return "Done!"

# Output when the module is loaded:
# Decorator called with function: slow_function
# Returning wrapper for slow_function

# 3. Call the decorated function
result = slow_function()
# Output when called:
# Wrapper called for slow_function
# Original function executing...
# Function slow_function took 1.00 seconds
```

#### Understanding the Closure

The wrapper function "remembers" the original function through a closure:

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # This wrapper function has access to 'func' from the outer scope
        # This is called a closure
        result = func(*args, **kwargs)  # 'func' is captured from outer scope
        return result
    return wrapper

@timer_decorator
def greet(name):
    return f"Hello, {name}!"

# The wrapper function still knows about the original 'greet' function
# even after timer_decorator has finished executing
```

#### Decorator Execution Order

When you stack multiple decorators, they execute from bottom to top:

```python
def bold(func):
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def hello(name):
    return f"Hello, {name}!"

# This is equivalent to:
# hello = bold(italic(hello))

print(hello("World"))
# Output: <b><i>Hello, World!</i></b>

# Execution order:
# 1. italic decorator is applied first: hello = italic(hello)
# 2. bold decorator is applied second: hello = bold(italic(hello))
# 3. When called: bold wrapper calls italic wrapper, which calls original hello
```

#### Why Use @wraps?

Without `@wraps`, the decorated function loses its original metadata:

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # ... timing code ...
        return func(*args, **kwargs)
    return wrapper

@timer_decorator
def slow_function():
    """This is the original docstring."""
    pass

print(slow_function.__name__)  # 'wrapper' (not 'slow_function')
print(slow_function.__doc__)   # None (lost the original docstring)

# With @wraps:
from functools import wraps

def timer_decorator(func):
    @wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs):
        # ... timing code ...
        return func(*args, **kwargs)
    return wrapper

@timer_decorator
def slow_function():
    """This is the original docstring."""
    pass

print(slow_function.__name__)  # 'slow_function' (preserved)
print(slow_function.__doc__)   # 'This is the original docstring.' (preserved)
```

#### Decorator with Parameters
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints "Hello, Alice!" three times
```

#### Multiple Decorators
```python
def bold(func):
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def hello(name):
    return f"Hello, {name}!"

print(hello("World"))  # Output: <b><i>Hello, World!</i></b>
```

#### Class-Based Decorators
```python
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Function {self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@Counter
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Function greet called 1 times
print(greet("Bob"))    # Function greet called 2 times
```

#### Practical Decorator Examples

##### 1. Logging Decorator
```python
import logging
from functools import wraps

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} raised an exception: {e}")
            raise
    return wrapper

@log_function
def divide(a, b):
    return a / b

# Setup logging
logging.basicConfig(level=logging.INFO)
divide(10, 2)  # Logs the function call and result
```

##### 2. Caching/Memoization Decorator
```python
from functools import wraps

def memoize(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This will be much faster due to caching
print(fibonacci(30))  # Cached result for subsequent calls
```

##### 3. Validation Decorator
```python
def validate_positive(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Argument {arg} must be positive")
        return func(*args, **kwargs)
    return wrapper

@validate_positive
def square_root(n):
    import math
    return math.sqrt(n)

print(square_root(4))  # 2.0
# print(square_root(-4))  # Raises ValueError
```

##### 4. Rate Limiting Decorator
```python
import time
from functools import wraps

def rate_limit(calls_per_second):
    def decorator(func):
        last_called = 0
        min_interval = 1.0 / calls_per_second
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_called
            current_time = time.time()
            time_since_last_call = current_time - last_called
            
            if time_since_last_call < min_interval:
                time.sleep(min_interval - time_since_last_call)
            
            last_called = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(2)  # Maximum 2 calls per second
def api_call():
    print("API call made")

# These calls will be rate-limited
for i in range(5):
    api_call()
```

##### 5. Authentication Decorator
```python
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Simulate checking authentication
        user_authenticated = check_authentication()
        if not user_authenticated:
            raise PermissionError("Authentication required")
        return func(*args, **kwargs)
    return wrapper

def check_authentication():
    # Simulate authentication check
    return True  # Change to False to test

@require_auth
def sensitive_operation():
    return "Sensitive data accessed"

print(sensitive_operation())  # Works if authenticated
```

##### 6. Retry Decorator
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def unreliable_function():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception("Random failure")
    return "Success!"

print(unreliable_function())  # Will retry up to 3 times
```

##### 7. Decorator with Function Metadata Preservation
```python
from functools import wraps

def preserve_metadata(func):
    @wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@preserve_metadata
def example_function():
    """This is the original function's docstring."""
    pass

# Without @wraps, this would show wrapper's metadata
print(example_function.__name__)  # 'example_function' (not 'wrapper')
print(example_function.__doc__)   # 'This is the original function's docstring.'
```

#### Decorator Best Practices

1. **Always use `@wraps`** to preserve function metadata
2. **Keep decorators simple** and focused on a single responsibility
3. **Handle exceptions appropriately** in decorators
4. **Consider performance implications** of decorators
5. **Use descriptive names** for decorator functions
6. **Document decorator behavior** clearly

#### Common Use Cases for Decorators

- **Logging and debugging**: Track function calls and execution time
- **Caching**: Store results of expensive computations
- **Validation**: Check input parameters before function execution
- **Authentication**: Verify user permissions
- **Rate limiting**: Control how often functions can be called
- **Error handling**: Implement retry logic or custom error handling
- **Performance monitoring**: Measure execution time and resource usage
- **API documentation**: Generate documentation from function signatures
