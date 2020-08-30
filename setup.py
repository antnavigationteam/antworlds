from distutils.core import setup
import os.path
from setuptools import setup, find_packages


def current_path(file_name):
    return os.abspath(os.path.join(__file__, os.path.pardir, file_name))


setup(
    name='antworlds',
    version='1.5',
    include_package_data=True,
    packages=find_packages(),
    license='BSD',
    author='Jan Stankiewicz',
    author_email='j.stankiewicz@ed.ac.uk',
    description='A module for simulating ants virtual scenes'
    )