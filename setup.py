# setup.py
from setuptools import setup, find_packages

setup(
    name='AirsProject',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.0',
        'torch>=1.10',
        'opencv-python',
        'numpy'
    ],
    author='你的名字',
    description='A YOLOv8-based object detection project.',
)
