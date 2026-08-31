This project is under WSL (Windows Subsystem for Linux) and is designed to run in a Linux environment.

Use bash script/build.sh to build the application or use WSL building tools.

Avoid unnecessary comments.

No over-engineering. Minimum complexity for current requirements. No premature abstractions, feature flags, or backward-compat shims.

When interacting with a running PoseWeaver instance, use the project-scoped PoseWeaver MCP tools instead of mouse, keyboard, screenshot, or computer-control automation. Start with `poseweaver_status`, launch with `poseweaver_launch` when needed, list assets before selecting by name, describe unfamiliar parameters before setting them, address layers by stable layer ID, and track long operations with the returned job ID until they reach a terminal state. Use `poseweaver_logs` for diagnostics. See `doc/automation-mcp.md` for the full workflow and direct bridge diagnostics.
