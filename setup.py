from setuptools import setup

setup(
  name = 'autoanalyzer',
  version = '0.1',
  author = 'Ha Hoang Hao',
  packages = ['Analyzer'],
  description = 'automatically analyze marketing metrics',
  install_requires = ['pandas', 'scikit-learn']
)
