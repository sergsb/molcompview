from setuptools import setup, find_packages

setup(
    python_requires=">=3.6",  # Your supported Python ranges
    name="molcompview",
    version="0.1.7",
    include_package_data=True,
    package_data={
        "molcompview": ["data/endocrine-receptor/*"],
    },
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "molcomplib",
        "rdkit",
        "appdata",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "fire",
        "dash",
        "dash-bootstrap-components",
    ],
    entry_points={
        "console_scripts": [
            "molcompview = molcompview.main:main",
        ],
    },
    license="MIT",
    author="Sergey Sosnin",
    author_email="sergey.sosnin@univie.ac.at",
    keywords="chemistry, cheminformatics, t-SNE, visualization,  chemical space",
    description="MolCompass Visualization Tool: Visualize your Chemical Space",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sergsb/molcompview",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Science/Research",
    ],
)
