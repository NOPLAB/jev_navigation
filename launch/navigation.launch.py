from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(Path(get_package_share_directory("jev_navigation")) / "config/navigation.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        Node(package="jev_navigation", executable="decision_node", name="decision_node",
             output="screen", parameters=[LaunchConfiguration("config"), {
                 "dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool)}],
             remappings=[("/camera/image_raw", LaunchConfiguration("image_topic")),
                         ("/odom", LaunchConfiguration("odom_topic")),
                         ("/cmd_vel", LaunchConfiguration("cmd_vel_topic"))]),
    ])
