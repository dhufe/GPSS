from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy 

ext_modules = [
    Extension(
        "gpss",
        ["gpss.pyx"],
        libraries=["m"],
        cython_directives={'language_level' : "3"},
        extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name = "gpss",
    cmdclass = {"build_ext": build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
