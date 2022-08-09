from pathlib import Path
from setuptools import setup, find_packages
import os


REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"

with open(str(REQUIREMENTS_PATH), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

os.system("chmod a+x init.sh examples/*")

setup(
    name="acleto",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "*.trainable_sampling*"]
    ),
    version="0.0.1",
    description="A Library for active learning. Supports text classification and sequence tagging tasks.",
    author="Tsvigun A., Sanochkin L., Kuzmin G., Larionov D., and Dr Shelmanov A.",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="NLP active AL deep learning transformer pytorch PLASM UPS",
    install_requires=requirements,
    include_package_data=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest==7.1.2"],
    test_suite="tests",
)

os.system("./init.sh")
