import traceback
def a():
    traceback.print_stack()
    print("a was called")
def b(): a()
def c(): a()
b()
