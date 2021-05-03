from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.0.4',
                     'h5py==2.7.0',
                     'tensorflow>=2.4.1',
                     'xgboost>=1.4.1'
                     ]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
