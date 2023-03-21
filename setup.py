#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="LightningDeepDTA",
    version="0.0.1",
    description="DeepDTA on PyTorchLightning with Hydra and cool stuff",
    author="Oleg Taratukhin",
    author_email="",
    url="https://github.com/olegtaratuhin/deep-dta-lightning",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
