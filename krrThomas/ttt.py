import numpy as np

def test(**kw):
    if kw:
        print('succes')

GSkwargs = {'r': 1}

print('without')
test()
print('with')
test(**GSkwargs)
