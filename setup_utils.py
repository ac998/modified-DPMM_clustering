#from distutils.core import setup
#from Cython.Build import cythonize

#setup(name="utils", ext_modules=cythonize('utils.pyx', annotate=True))

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("utils",
              ["utils.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "utils",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)