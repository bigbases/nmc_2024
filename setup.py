from setuptools import setup, find_packages

setup(
    name='nmc',
    version='0.0.1',
    description='nactional medical center diabetic retinopathy project ',
    url='https://github.com/bigbases/nmc_2024',
    author='Gangmin park, Wonryeol jeong, Jihoon moon',
    author_email='gangmin.park@seoultech.ac.kr',
    license='bigbase',
    packages=find_packages(include=['nmc']),
    install_requires=[
        'tqdm',
        'tabulate',
        'numpy',
        'scipy',
        'matplotlib',
    ]
)