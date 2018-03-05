from test import test as test_py
from test_cy import test as test_cy
import timeit

py = timeit.timeit('test.test(1000)', setup='import test', number=100)
cy = timeit.timeit('test_cy.test(1000)', setup='import test_cy', number=100)

print('test_py(1000)=', test_py(1000))
print('test_cy(1000)=', test_cy(1000))

print('py runtime:', py)
print('cy runtime:', cy)
print('times faster:', py/cy)
