from setuptools import setup, find_packages

setup(name='baffle',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='Confusion metrics for machine learning',
      author='Walter Hoekstra',
      author_email='walter@newzoo.com',
      license='MIT',
      install_requires=[
		  'pandas'],
      zip_safe=False)