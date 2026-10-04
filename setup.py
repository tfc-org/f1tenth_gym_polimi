from glob import glob

from setuptools import find_packages, setup

package_name = 'polimi_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TF-Costantini',
    maintainer_email='tommasofabrizio.costantini@mail.polimi.it',
    description='Car-faithful layer on top of f1tenth_gym_ros.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_model = polimi_sim.lidar_model_node:main',
            'actuation_model = polimi_sim.actuation_model_node:main',
            'odom_model = polimi_sim.odom_model_node:main',
            'teleop_bridge = polimi_sim.teleop_bridge_node:main',
        ],
    },
)
