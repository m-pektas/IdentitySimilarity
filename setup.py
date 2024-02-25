
from setuptools import setup, find_packages
import os

def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        return f.read()

def get_requirements(req_path: str):
    with open(req_path, encoding='utf8') as f:
        return f.read().splitlines()



__version__ = "0.0.1"
__author__ = "Muhammed Pektas"

setup(
     name='idsim',  
     version=__version__,
     author=__author__,
     author_email="mhmdpkts@gmail.com",
     description="Face Recognition Metric",
     long_description=get_long_description(),
     long_description_content_type="text/markdown",
     url="https://github.com/m-pektas/IdentitySimilarity",
     packages=find_packages(),
     install_requires=get_requirements("requirements.txt"),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
     python_requires=">=3.7",
 )
