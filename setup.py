from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['numpy>=1.14.6', 'tensorflow==2.6.0', 'scipy>=1.1.0', 'scikit-learn>=1.0', 'keras==2.6.0']
setup(
    name='trainerr',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classifier test'
)