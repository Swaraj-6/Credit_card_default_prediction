from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        requirements = [req.strip() for req in requirements if req.strip()!=HYPEN_E_DOT]

    return requirements

setup(
    name="Credit-Card-Project",
    version="0.0.1",
    author="Swaraj",
    author_email="swarajkakade83@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)


