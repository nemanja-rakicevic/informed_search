
import os
from setuptools import setup, find_packages, Command


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
    # Needed to silence warnings (and to be a worthwhile package)
    name='InformedSearch',
    url='https://github.com/nemanja-rakicevic/informed_search',
    author='Nemanja Rakicevic',
    author_email='n.rakicevic@imperial.ac.uk',
    # Needed to actually package something
    # packages=[package for package in find_packages() if package.startswith('informed_search')],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Code for Informed Search in discrete parameter space',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
    cmdclass={
        'clean': CleanCommand,
    },
    extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']
)
