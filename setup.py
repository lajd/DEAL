import os
from typing import List
from setuptools import setup, find_packages


def get_requirements() -> List[str]:
    with open(os.path.join('.', 'requirements.txt'), 'r') as f:
        requirements = [i.strip() for i in f.readlines() if not i.startswith('#')]
    return requirements


setup(
    name='DEAL',
    version="0.1.0",
    platforms=['any'],
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.9',
    include_package_data=True,
    install_requires=get_requirements(),
    package_data={},
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
