import setuptools

setuptools.setup(
    name="hello-world-cli",
    version="1.0.0",
    author="Your Name",
    author_email="your@email.com",
    description="A simple CLI tool that prints a greeting message.",
    packages=setuptools.find_packages(),
    install_requires=["click==8.1.3"],
    entry_points={
        "console_scripts": [
            "hello-world-cli=main:main",
        ],
    },
)