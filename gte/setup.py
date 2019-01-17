from setuptools import setup, find_packages

setup(
    name='gte',
    version='0.1',
    description='Grounded Textual Entailment',
    url='https://github.com/AlessandroSteri/computer_vision.git',
    author='Alessandro Steri & Agostina Calabrese',
    # author_email='',
    license='MIT',
    packages=find_packages(exclude=['docs', 'tests']),
    # install_requires=['pickle', ],
    # dependency_links=[''],
    zip_safe=False
    )
