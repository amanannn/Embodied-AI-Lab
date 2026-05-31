# sim2d

中文： [README.md](./README.md)

This module will host lightweight two-dimensional simulation infrastructure extracted from course experiments.

## Intended Ownership

- common world stepping logic
- robot state update helpers
- sensor simulation hooks
- obstacle and map primitives used by multiple directions

## Out of Scope

The following does not belong here:

- one-off scenario logic for a single navigation or control lab
- experiment-specific reward shaping or curriculum code
- simulator wrappers whose API is still changing as part of one direction's lesson flow
