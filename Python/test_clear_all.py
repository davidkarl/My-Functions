
__saved_context__ = {}

def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

saveContext()

hello = 'hi there'
print hello             # prints "hi there" on stdout

restoreContext()

print hello             # throws an exception



