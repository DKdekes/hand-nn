import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hand-nn",
    version="0.0.1",
    author="dkdekes",
    description="A small neural network package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DKdekes/hand-nn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)