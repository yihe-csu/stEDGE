#!/usr/bin/env python
"""Package configuration for stEDGE."""

from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def read_requirements(path="requirements.txt"):
    requirements = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


def read_version(path="stEDGE/__init__.py"):
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', read_text(path))
    if not match:
        raise RuntimeError("Unable to find __version__ in stEDGE/__init__.py")
    return match.group(1).strip()


setup(
    name="stEDGE",
    version=read_version(),
    author="Yi He",
    author_email="yihe_csu@csu.edu.cn",
    description=(
        "Edge-guided multiscale reconstruction of hierarchical spatial domains "
        "and transition interfaces in spatial transcriptomics."
    ),
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/yihe-csu/stEDGE",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.10",
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
