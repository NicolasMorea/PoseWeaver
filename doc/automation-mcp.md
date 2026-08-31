# PoseWeaver automation and MCP integration

PoseWeaver exposes a typed local automation API so agents can operate the running application without desktop-control automation. All reads and mutations are executed by PoseWeaver's main thread; the socket listener never touches `AppState`, canvas pixels, ImGui, or OpenGL state directly.

## Architecture

```text
Codex MCP client
  -> scripts/poseweaver_mcp.py (MCP over stdio)
  -> private Unix socket (newline-delimited JSON)
  -> AutomationServer command queue
  -> PoseWeaver main/render thread
```

The socket defaults to `/tmp/poseweaver-$UID.sock`, has mode `0600`, and can be overridden with `POSEWEAVER_AUTOMATION_SOCKET`. The MCP process and PoseWeaver must use the same value.

The project-scoped `.codex/config.toml` starts the MCP adapter through WSL. Restart the Codex client after first adding or changing an MCP configuration so it can refresh the tool inventory.

## Agent workflow

1. Call `poseweaver_status`.
2. If it is not running, call `poseweaver_launch`.
3. Call `poseweaver_assets` before selecting asset names.
4. Call `poseweaver_describe_parameters` before changing unfamiliar parameters.
5. Use stable layer IDs returned by `poseweaver_list_layers`.
6. Start pipelines with `poseweaver_start_operation`.
7. Poll `poseweaver_get_job` until its state is `completed`, `failed`, or `cancelled`.
8. Read `poseweaver_logs` when an operation fails or produces an unexpected result.

Do not use mouse, keyboard, browser, screenshot, or computer-control tools for behavior covered by the PoseWeaver MCP tools.

## Available operations

- `simulation`
- `generate_cloth`
- `full_generation`
- `generate_and_fit`
- `render_cloth` (supports `fine`)
- `cloth_fitting`
- `extract_pose`
- `rig_optimization`
- `generate_displacement`
- `generate_iso_displacement`
- `scene_snapshot`
- `garment_pushout`, `garment_ease`, `garment_prefit`, `garment_relax`
- `save_constraints`, `clear_constraints`, `clear_simulation`
- `undo_canvas`, `reset_canvas`

Only simulation jobs support force-free cancellation. Other pipelines share long-lived inference workers and are allowed to finish so their state cannot be left partially mutated.

## Direct bridge diagnostics

The internal bridge can be inspected without an MCP client:

```bash
python3 scripts/poseweaver_mcp.py --bridge-call app.status
python3 scripts/poseweaver_mcp.py --bridge-call parameters.get --params '{"names":["simulation.gravity"]}'
```

The direct bridge is deliberately application-specific. It does not expose shell execution or arbitrary process creation.

## Verification

```bash
scripts/build.sh
python3 -m unittest tests/test_poseweaver_mcp.py
```
