from setuptools import setup, find_packages

setup(
    name="topological_hough_transform",
    version="0.1.0",  # Initial version
    author="Martin Uray",
    author_email="martin.uray@fh-salzburg.ac.at",
    description="A module implementing a topological variant of the Hough "
                "Transform for detecting lines in images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JRC-ISIA/TopologicalHoughTransformation",
    packages=find_packages(),  # Automatically find submodules
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
