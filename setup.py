from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

requirements = [line.strip() for line in open("./requirements.txt", "r")]

setup(
    name="diffmpc",
    version="0.0.1",
    author="diffmpc",
    author_email="diffmpc@example.com",
    description="Differentiable optimal control package with GPU support.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://example.com",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.10",
)
