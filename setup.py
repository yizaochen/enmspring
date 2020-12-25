from setuptools import setup, find_packages

setup(name='enmspring', 
      version='0.1',
      packages=find_packages(),
      url='https://github.com/yizaochen/enmspring.git',
      author='Yizao Chen',
      author_email='yizaochen@gmail.com',
      install_requires=[
          'MDAnalysis',
          'matplotlib',
          'pandas',
          'scipy',
          'seaborn',
          'networkx'
      ]
      )