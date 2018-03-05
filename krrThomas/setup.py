from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('featureCalculators/angular_fingerprintFeature_cy.pyx'))
