from setuptools import setup

VERSION = '0.0.1'

setup(
    name='dact-qualkit',
    version=VERSION,
    py_modules=['dact-qualkit'],
    url='https://github.com/scottbw/suda',
    download_url='https://github.com/scottbw/suda/tarball/{}'.format(VERSION),
    license='MIT',
    author='Scott Wilson',
    author_email='scott.bradley.wilson@gmail.com',
    description='Python library for implementing Special Uniques Detection Algorithm (SUDA) '
                'for measuring disclosure control risk in synthetic data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['pandas~=1.1.5'],
    entry_points={
        'console_scripts': ['suda=suda:main'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)