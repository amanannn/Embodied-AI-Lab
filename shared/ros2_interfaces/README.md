# ros2_interfaces

This module will define the boundary between the educational experiments and ROS2-facing integration work.

## Intended Ownership

- thin wrappers around ROS2 integration points
- shared conventions for publishers, subscribers, or messages
- bridge code used by mixed Python and C++ direction labs

## Out of Scope

The following does not belong here:

- full application nodes owned by one capstone or direction
- hardware bring-up notes for a single robot platform
- experiment-local callbacks or launch setups that are not reused outside one lab
