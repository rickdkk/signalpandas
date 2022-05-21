from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(
        name="signalpandas",
        version="0.1.3",
        author="Rick de Klerk",
        author_email="rickdkk@gmail.com",
        description="Bringing signal analysis to Pandas!",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/rickdkk/signalpandas",
        project_urls={
            "Bug Tracker": "https://github.com/rickdkk/signalpandas/-/issues",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "."},
        packages=find_packages(where="signalpandas"),
        python_requires=">=3.6",
    )
