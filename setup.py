# coding=utf-8
# email: wangzejunscut@126.com

import io
from setuptools import setup, find_packages

def _get_readme():
    """
    Use pandoc to generate rst from md.
    pandoc --from=markdown --to=rst --output=README.rst README.md
    """
    with io.open("README.rst", encoding='utf-8') as fid:
        return fid.read()


setup(
    name='gpt2ppl-zh',
    version='0.3.0',
    author='wangzejun',
    author_email='wangzejunscut@126.com',
    description='Chinese sentence perplexity calculation based on GPT2 pre-trained model',
    long_description=_get_readme(),
    license='MIT License',
    url='https://github.com/zejunwang1/gpt2ppl-zh',
    install_requires=['transformers', 'torch'],
    packages=find_packages()
)
