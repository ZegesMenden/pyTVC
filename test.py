def makeReg():

    registry = {}

    def reg(func):
        registry[func.__name__] = func
        return func

    reg.all = registry
    return reg

r = makeReg()

@r
def test1():
    print("test1")

@r
def test2():
    print("test2")

print(r.all)
for i in r.all:
    r.all[i]()