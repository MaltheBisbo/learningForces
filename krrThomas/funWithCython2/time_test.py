
from test_cy import test_bondtypes as test
test()

"""
from test import test as test_py
from test_cy import test as test_cy
import timeit

py = timeit.timeit('test.test(20)', setup='import test', number=100)
cy = timeit.timeit('test_cy.test(20)', setup='import test_cy', number=100)

print('test_py(1000)=', test_py(20))
print('test_cy(1000)=', test_cy(20))

print('py runtime:', py)
print('cy runtime:', cy)
print('times faster:', py/cy)
"""
