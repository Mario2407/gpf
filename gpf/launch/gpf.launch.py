"""Launch Geometric Primitive Fitting"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir


def generate_launch_description():
    config_gpf = LaunchConfiguration('config_gpf', default=os.path.join(get_package_share_directory(
        'gpf'), 'config', 'gpf.yaml'))
    config_rviz2 = LaunchConfiguration('config_rviz2', default=os.path.join(get_package_share_directory(
        'gpf'), 'config', 'rviz2.rviz'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_gpf',
            default_value=config_gpf,
            description='Path to config for GPF'),
        DeclareLaunchArgument(
            'config_rviz2',
            default_value=config_rviz2,
            description='Path to config for RViz2'),

        Node(
            package='gpf',
            executable='gpf',
            name='gpf',
            namespace='',
            output='screen',
            parameters=[config_gpf],
            remappings=[
                        ('camera/pointcloud', '/zed2i_top/zed2i/point_cloud/cloud_registered'),
                        # ('camera/pointcloud', '/livox/lidar'),
                        # ('camera/pointcloud', '/livox/lidar/concatenated'),
                        ('visualisation_markers', 'gpf/visualisation_markers'),
                        ('geometric_primitives', 'gpf/geometric_primitives')],
        ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         [os.path.join(get_package_share_directory('gpf'), 'launch',
        #                       'rviz2.launch.py')]),
        #     launch_arguments=[('config_rviz2', config_rviz2)]
        # ),
    ])
