# setup.py
from setuptools import setup, find_packages

setup(
    name='AirsProject',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'torch',
        'opencv-python',
        'numpy'
    ],
    author='Hzp1231',
    description='A YOLOv8-based object detection project.',
)
