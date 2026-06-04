# ros2_interfaces — ROS2 Bridge Interface Contract

中文： [README.md](./README.md)

## Positioning

This module is the Level 1 → Level 2 ROS2 bridge interface layer. It defines messages and conventions between Python experiment outputs and ROS2 systems, without containing full application nodes or hardware bring-up logic.

## Role in the Python-first, ROS2-ready Path

```
Level 1 (Python)          Level 2 (ROS2/C++)
    ↓                         ↑
    └── ros2_interfaces ──────┘
         Message definitions + bridge conventions
```

- Level 1 Python experiments output data in standard formats (CSV, JSON, numpy arrays)
- `ros2_interfaces` defines how this data maps to ROS2 messages
- Level 2 C++ / ROS2 experiments consume these messages to connect into real robot software stacks

## Current State

This module is currently in interface planning stage. Candidate contents include:

- Custom ROS2 message definitions (`.msg` files)
- Service definitions (`.srv` files)
- Python ↔ ROS2 bridge helper scripts
- Mapping documentation between messages and Level 1 output formats

## Interface Conventions

### Level 1 Output Formats

| Direction | Typical Output | Format |
|-----------|---------------|--------|
| Perception | State estimation, sensor fusion | CSV / numpy arrays |
| SLAM and Navigation | Paths, poses, maps | CSV / JSON |
| Motion Control | Trajectories, control signals | CSV / numpy arrays |
| RL | Policies, actions | numpy arrays |

### ROS2 Message Mapping

Candidate message definitions will follow these principles:

- Maintain direct correspondence with Level 1 output formats
- Do not introduce extra fields not needed by Level 1
- Prefer standard ROS2 message types (`geometry_msgs`, `sensor_msgs`, `nav_msgs`)
- Custom messages only when standard types cannot cover the use case

## Out of Scope

- Full application nodes owned by one capstone or direction
- Hardware bring-up notes for a single robot platform
- Experiment-local callbacks or launch setups that are not reused outside one lab

## Related Documentation

- [Python-first, ROS2-ready Path](../../docs/curriculum/python-first-ros2-ready.en.md)
- [Legacy-to-Direction Mapping](../../docs/curriculum/legacy-to-direction-map.en.md)
