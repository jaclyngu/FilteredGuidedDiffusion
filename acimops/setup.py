import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import sys

if sys.platform == "win32":
    ext_mods = [ Extension("cbilateral",
                           ["acimops/basic_bilateral_filter.pyx"]
                           )]
else:
    ext_mods = [ Extension("cbilateral",
                           ["acimops/basic_bilateral_filter.pyx"],
                           # Compiling on Windows, we have commented out the next two lines.
                           libraries=["m"],
                           extra_compile_args = ["-ffast-math"] # This will work on gcc; other C compilers would have different flags, but this isn't required. Visual C will not recognize it.
                           )]


with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acimops",
    version="0.0.1",
    author="   ", # you can change this to you
    description="cython image ops for py.Image",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.coecis.cornell.edu/CS6682-Spring2021/a2_bilateral/",
    project_urls={
        'My webpage': 'yourwebpage.com',
        'Source': 'https://github.coecis.cornell.edu/-------',
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'cython'
    ],
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests*']),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules = cythonize(ext_mods),
    zip_safe=False,
)
