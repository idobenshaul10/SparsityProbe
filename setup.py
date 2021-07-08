from setuptools import setup, find_packages

setup(
	name='Probe',
	version='1.0',
	description='SparsityProbe',
	author='ido',
	author_email='ido.benshaul@gmail.com',
	url='',
	packages =find_packages(include=['DL_Layer_Analysis', 'DL_Layer_Analysis.*'])
)