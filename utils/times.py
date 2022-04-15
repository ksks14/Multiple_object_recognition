def func_time(func):
    """
    利用装饰器计算方法或者函数的时间
    :param func:
    :return:
    """
    from time import time

    def init_func(*args, **kwargs):
        """
        内部函数
        :param args:
        :param kwargs:
        :return:
        """
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        print(f'{func.__name__}:time:', end - start,'s')
        return func_return

    return init_func
