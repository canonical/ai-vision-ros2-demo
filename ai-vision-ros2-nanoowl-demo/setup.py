from glob import glob

from setuptools import find_packages, setup

package_name = "ai_vision_ros2_nanoowl_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Giuseppe Barbieri",
    maintainer_email="giuseppe.barbieri@canonical.com",
    description="NanoOWL (OWL-ViT) open-vocabulary detection demo for SO-101 on NVIDIA Jetson (TensorRT)",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "owl_detect_node = ai_vision_ros2_nanoowl_demo.owl_detect_node:main",
        ],
    },
)
