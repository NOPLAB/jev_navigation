from glob import glob
from setuptools import setup

setup(
    name="jev_navigation",
    version="0.1.0",
    packages=["jev_navigation"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/jev_navigation"]),
        ("share/jev_navigation", ["package.xml", "LICENSE"]),
        ("share/jev_navigation/launch", glob("launch/*.launch.py")),
        ("share/jev_navigation/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nop",
    maintainer_email="noplab90@gmail.com",
    author="nop",
    author_email="noplab90@gmail.com",
    description="Image-based action selection using decider-2b-vision",
    license="BSD-3-Clause",
    entry_points={"console_scripts": ["decision_node = jev_navigation.decision_node:main"]},
)
