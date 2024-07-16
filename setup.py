from setuptools import setup, find_packages


setup(
    name='baba',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('baba')],
    install_requires=[
        "gym<=0.25.0",
        "numpy>=1.18.0",
        "matplotlib>=3.0",
        "opencv-python",
        "typing_extensions",
        "pygame"
    ],
    description='',
    author=''
)
