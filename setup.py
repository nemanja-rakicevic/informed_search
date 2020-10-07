
import os
from setuptools import setup, Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='informed_search',
    url='https://github.com/nemanja-rakicevic/informed_search',
    author='Nemanja Rakicevic',
    author_email='n.rakicevic@imperial.ac.uk',
    version='0.1',
    license='MIT',
    description='Code for Informed Search in discrete parameter space',
    long_description=open('README.md').read(),
    cmdclass={
        'clean': CleanCommand,
    },
    extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']
)
