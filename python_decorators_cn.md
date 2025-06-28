# Python 装饰器：综合指南

Python 装饰器是一个函数，它可以在不改变另一个函数（或类/方法）源代码的情况下修改其行为。装饰器通常用于日志记录、计时、访问控制、记忆化等。在本教程中，我们将学习装饰器的工作原理，以及如何在不同应用中有效地创建和使用它们。

#### 基本装饰器语法
```python
def timer_decorator(func):
    from time import time
    
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"函数 {func.__name__} 执行了 {end - start:.2f} 秒")
        return result
    
    return wrapper

@timer_decorator
def slow_function():
    import time
    time.sleep(1)
    return "完成！"

slow_function()
```
结果是：
```python
>>> slow_function()
函数 slow_function 执行了 1.00 秒
'完成！'
```

#### 装饰器实际工作原理

`@timer_decorator` 语法实际上是语法糖。以下是逐步发生的事情：

**步骤 1：装饰器定义**
```python
def timer_decorator(func):
    # 这个函数接受另一个函数作为参数
    # 并返回一个新函数（包装器）
    
    def wrapper(*args, **kwargs):
        # 这是将替换原始函数的新函数
        start = time()
        result = func(*args, **kwargs)  # 调用原始函数
        end = time()
        print(f"函数 {func.__name__} 执行了 {end - start:.2f} 秒")
        return result
    
    return wrapper  # 返回包装器函数
```

**步骤 2：@timer_decorator 的作用**
`@timer_decorator` 语法等同于：
```python
def slow_function():
    import time
    time.sleep(1)
    return "完成！"

# 这行代码与 @timer_decorator 做同样的事情
slow_function = timer_decorator(slow_function)
```

**步骤 3：转换过程**
```python
# 装饰前
def slow_function():
    import time
    time.sleep(1)
    return "完成！"

# 装饰后（实际发生的事情）
def slow_function():
    # 现在这是包装器函数
    start = time()
    result = original_slow_function()  # 调用原始函数
    end = time()
    print(f"函数 slow_function 执行了 {end - start:.2f} 秒")
    return result
```

**步骤 4：函数调用流程**
```python
# 当你调用 slow_function() 时，发生以下事情：
slow_function()
# ↓
wrapper()  # 调用包装器函数
# ↓
start = time()
result = original_slow_function()  # 原始函数执行
end = time()
print(f"函数 slow_function 执行了 {end - start:.2f} 秒")
return result
```

#### 装饰器执行的可视化表示

```python
# 1. 定义装饰器
def timer_decorator(func):
    print(f"装饰器被调用，函数：{func.__name__}")
    
    def wrapper(*args, **kwargs):
        print(f"为 {func.__name__} 调用包装器")
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"函数 {func.__name__} 执行了 {end - start:.2f} 秒")
        return result
    
    print(f"为 {func.__name__} 返回包装器")
    return wrapper

# 2. 应用装饰器
@timer_decorator
def slow_function():
    print("原始函数执行中...")
    time.sleep(1)
    return "完成！"

# 模块加载时的输出：
# 装饰器被调用，函数：slow_function
# 为 slow_function 返回包装器

# 3. 调用装饰后的函数
result = slow_function()
# 调用时的输出：
# 为 slow_function 调用包装器
# 原始函数执行中...
# 函数 slow_function 执行了 1.00 秒
```

#### 理解闭包

包装器函数通过闭包"记住"原始函数：

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # 这个包装器函数可以访问外部作用域的 'func'
        # 这被称为闭包
        result = func(*args, **kwargs)  # 'func' 从外部作用域捕获
        return result
    return wrapper

@timer_decorator
def greet(name):
    return f"你好，{name}！"

# 包装器函数仍然知道原始的 'greet' 函数
# 即使在 timer_decorator 执行完毕后
```

#### 装饰器执行顺序

当你堆叠多个装饰器时，它们从下到上执行：

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
    return f"你好，{name}！"

# 这等同于：
# hello = bold(italic(hello))

print(hello("世界"))
# 输出：<b><i>你好，世界！</i></b>

# 执行顺序：
# 1. italic 装饰器首先应用：hello = italic(hello)
# 2. bold 装饰器其次应用：hello = bold(italic(hello))
# 3. 调用时：bold 包装器调用 italic 包装器，后者调用原始 hello
```

#### 为什么使用 @wraps？

没有 `@wraps`，装饰后的函数会失去其原始元数据：

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # ... 计时代码 ...
        return func(*args, **kwargs)
    return wrapper

@timer_decorator
def slow_function():
    """这是原始文档字符串。"""
    pass

print(slow_function.__name__)  # 'wrapper'（不是 'slow_function'）
print(slow_function.__doc__)   # None（丢失了原始文档字符串）

# 使用 @wraps：
from functools import wraps

def timer_decorator(func):
    @wraps(func)  # 这保留了原始函数的元数据
    def wrapper(*args, **kwargs):
        # ... 计时代码 ...
        return func(*args, **kwargs)
    return wrapper

@timer_decorator
def slow_function():
    """这是原始文档字符串。"""
    pass

print(slow_function.__name__)  # 'slow_function'（保留）
print(slow_function.__doc__)   # '这是原始文档字符串。'（保留）
```

#### 带参数的装饰器
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
    print(f"你好，{name}！")

greet("爱丽丝")  # 打印三次 "你好，爱丽丝！"
```

#### 多个装饰器
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
    return f"你好，{name}！"

print(hello("世界"))  # 输出：<b><i>你好，世界！</i></b>
```

#### 基于类的装饰器
```python
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"函数 {self.func.__name__} 被调用了 {self.count} 次")
        return self.func(*args, **kwargs)

@Counter
def greet(name):
    return f"你好，{name}！"

print(greet("爱丽丝"))  # 函数 greet 被调用了 1 次
print(greet("鲍勃"))    # 函数 greet 被调用了 2 次
```

#### 实用装饰器示例

##### 1. 日志装饰器
```python
import logging
from functools import wraps

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"调用 {func.__name__}，参数：{args}，关键字参数：{kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} 返回：{result}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} 抛出异常：{e}")
            raise
    return wrapper

@log_function
def divide(a, b):
    return a / b

# 设置日志
logging.basicConfig(level=logging.INFO)
divide(10, 2)  # 记录函数调用和结果
```

##### 2. 缓存/记忆化装饰器
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

# 由于缓存，这将快得多
print(fibonacci(30))  # 后续调用的缓存结果
```

##### 3. 验证装饰器
```python
def validate_positive(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"参数 {arg} 必须为正数")
        return func(*args, **kwargs)
    return wrapper

@validate_positive
def square_root(n):
    import math
    return math.sqrt(n)

print(square_root(4))  # 2.0
# print(square_root(-4))  # 抛出 ValueError
```

##### 4. 速率限制装饰器
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

@rate_limit(2)  # 每秒最多 2 次调用
def api_call():
    print("API 调用已执行")

# 这些调用将被速率限制
for i in range(5):
    api_call()
```

##### 5. 认证装饰器
```python
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 模拟检查认证
        user_authenticated = check_authentication()
        if not user_authenticated:
            raise PermissionError("需要认证")
        return func(*args, **kwargs)
    return wrapper

def check_authentication():
    # 模拟认证检查
    return True  # 改为 False 进行测试

@require_auth
def sensitive_operation():
    return "敏感数据已访问"

print(sensitive_operation())  # 如果已认证则工作
```

##### 6. 重试装饰器
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
                    print(f"尝试 {attempt + 1} 失败：{e}。重试中...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def unreliable_function():
    import random
    if random.random() < 0.7:  # 70% 失败概率
        raise Exception("随机失败")
    return "成功！"

print(unreliable_function())  # 将重试最多 3 次
```

##### 7. 保留函数元数据的装饰器
```python
from functools import wraps

def preserve_metadata(func):
    @wraps(func)  # 这保留了原始函数的元数据
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@preserve_metadata
def example_function():
    """这是原始函数的文档字符串。"""
    pass

# 没有 @wraps，这会显示包装器的元数据
print(example_function.__name__)  # 'example_function'（不是 'wrapper'）
print(example_function.__doc__)   # '这是原始函数的文档字符串。'
```

#### 装饰器最佳实践

1. **始终使用 `@wraps`** 来保留函数元数据
2. **保持装饰器简单** 并专注于单一职责
3. **适当处理异常** 在装饰器中
4. **考虑装饰器的性能影响**
5. **使用描述性名称** 为装饰器函数
6. **清楚地记录装饰器行为**

#### 装饰器的常见用例

- **日志记录和调试**：跟踪函数调用和执行时间
- **缓存**：存储昂贵计算的结果
- **验证**：在函数执行前检查输入参数
- **认证**：验证用户权限
- **速率限制**：控制函数可以被调用的频率
- **错误处理**：实现重试逻辑或自定义错误处理
- **性能监控**：测量执行时间和资源使用情况
- **API 文档**：从函数签名生成文档 