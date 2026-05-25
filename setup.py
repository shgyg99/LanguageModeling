from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Language Modeling",
    version="0.1",
    author="Shgyg",
    packages=find_packages(),
    install_requires=requirements,
)


# pip install -e .
