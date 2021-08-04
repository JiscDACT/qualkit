from setuptools import setup

VERSION = '0.0.9'

setup(
    name='qualkit',
    version=VERSION,
    packages=['qualkit'],
    url='https://github.com/JiscDACT/qualkit',
    download_url='https://github.com/JiscDACT/qualkit/tarball/{}'.format(VERSION),
    license='BSD',
    author='Scott Wilson',
    author_email='scott.wilson@jisc.ac.uk',
    description='Python qualitative analysis toolkit with utilities and simplified wrappers for common algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['pandas>=1.1.5',
                        'numpy>=1.19.5',
                        'nltk>=3.6.2',
                        'scikit-learn>=0.24.2',
                        'corextopic>=1.1'],
    entry_points={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)