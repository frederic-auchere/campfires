# To install: pip install .

from setuptools import setup, find_packages

NAME = 'campfires'
DESCRIPTION = 'EUI campfire detection',
EMAIL = 'frederic.auchere@universite-paris-saclay.fr'
AUTHOR = 'Frédéric Auchère'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'

entry_points = {
    'console_scripts': [
        'campfires=src.cli:cli',
        ]
    }

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().strip('\n').split('\n')

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url='https://github.com/frederic-auchere/campfires',
    install_requires=requirements,
    license='LGPL-v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    zip_safe=False,
    ext_modules=None,
    entry_points=entry_points
)
