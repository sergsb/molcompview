from setuptools import setup,find_packages

setup(
    # Your setup arguments
    python_requires='>=3.6',  # Your supported Python ranges
    name = "molcompview",
    version = "0.1.0",
    description = "MolCompass Viewer",
    author = "Sergey Sosnin <serg.sosnin@univie.ac.at>",
    include_package_data=False,
    packages=find_packages(),
    install_requires=[
        'molcomplib',
        'rdkit',
        'numpy',
        'fire',
        'dash',
        'dash-bootstrap-components',
    ],
    packages=["molcompview"],
    entry_points={
        'console_scripts': [
            'molcompview = molcompview.main:entry_point',
        ],
    },
    license = "MIT",
    author='Sergey Sosnin',
    author_email='sergey.sosnin@univie.ac.at',
    description='MolCompass visualization tool: visualize your chemical space',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sergsb/molcompview',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
