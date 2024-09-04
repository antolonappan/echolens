from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="echolens",
    version="0.1",
    packages=find_packages(include=['echolens', 'echolens.*']),
    include_package_data=True,
    description="A package for CMB simulations and instrument modeling",
    maintainer="Anto Idicherian Lonappan",
    maintainer_email="mail@antolonappan.me",  
    package_data={
        'echolens.simulation': ['echo.ini', 'masks.fits', 'spectra.pkl'], 
    },
)