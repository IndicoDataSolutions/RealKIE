from setuptools import find_packages, setup

setup(
    name="metrics",
    version="0.1",
    packages=find_packages(),
    install_requires=["fire", "numpy", "pandas",],
    entry_points="""
        [console_scripts]
        metrics=metrics.metrics:cli
    """,
)
