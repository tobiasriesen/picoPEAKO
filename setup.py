from setuptools import setup, find_packages

setup(
    name='picopeako',
    version='0.1.0',
    author='Tobias Riesen',
    author_email='tobias.riesen@gmail.com',
    description='A Python library for radar data processing and analysis using scipy, numpy, and xarray.',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'xarray',
        'matplotlib',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
