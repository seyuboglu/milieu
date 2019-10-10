"""
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    entry_points={
        'console_scripts':[
            'run=milieu.run:main'
        ]
    },
    name="milieu",
    version="0.0.1",
    author="Evan Sabri Eyuboglu",
    author_email="eyuboglu@stanford.edu",
    description="Research software for Stanford SNAP disease protein prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/milieu",
    packages=setuptools.find_packages(include=['milieu']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'tqdm', 'click', 'matplotlib', 'networkx', 'ndex2', 'cyjupyter',
        'goatools', 'parse', 'seaborn', 'jupyter', 'ipywidgets', 'ipykernel',
        'visJS2jupyter'
    ]
)