from setuptools import setup, find_packages

setup(
    name="rl2023",
    version="0.1",
    description="Reinforcement Learning in UoE (CW)",
    author="Keith Ciantar ",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "numpy>=1.18",
        "torch>=1.3",
        "gym>=0.12,<0.26",
        "gym[box2d]",
        "tqdm>=4.41",
        "pyglet>=1.3",
        "matplotlib>=3.1",
        "pytest>=5.3",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
