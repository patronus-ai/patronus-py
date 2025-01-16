from patronus.tracing import init, get_logger
from patronus.tracing.decorators import traced

"""
export PATRONUS_API_KEY='<your api key>'
"""

# Initialize Patronus Client

init("MyDemoProject")


@traced()
def my_func_2(foo: str, foo_with_defult: str = "default `foo`", **kwargs):
    return foo

@traced()
def my_func_3(*args, **kwargs):
    my_func_2(foo='test')
    return "bar"

# my_func_3(1, 2, 3, kwarg1="kwarg1val")


logger = get_logger("MyDemoProject")
logger.debug("Hello World debug")
logger.error("Hello World debug")
# logger.evaluation_log() - To be implemented