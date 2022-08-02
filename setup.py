
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='idsim',  
     version='0.0.1',
     author="Muhammed Pektas",
     author_email="mhmdpkts@gmail.com",
     description="Face Recognition Metric",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/m-pektas/IdentitySimilarity",
     packages=find_packages(),
     
     
    #  package_data={'idsim': ['models/*.pth']},
    #  include_package_data=True,
     install_requires=["torch>=1.7.0", "torchvision>=0.8.1", "numpy>=1.14.3", "scikit-image>=0.19.2", "munch", "kornia==0.6.4", "gdown"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
     include_dirs=["models"]
 )