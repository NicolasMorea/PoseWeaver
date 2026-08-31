#!/usr/bin/env python3
"""Dependency-free MCP adapter for controlling a running PoseWeaver instance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build" / "poseweaver"
DEFAULT_LOG = ROOT / "build" / "poseweaver-automation.log"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


class BridgeError(RuntimeError):
    pass


def default_socket_path() -> Path:
    configured = os.environ.get("POSEWEAVER_AUTOMATION_SOCKET")
    return Path(configured) if configured else Path(f"/tmp/poseweaver-{os.getuid()}.sock")


class BridgeClient:
    def __init__(self, socket_path: Path | None = None, timeout: float = 20.0):
        self.socket_path = socket_path or default_socket_path()
        self.timeout = timeout
        self._next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        request = json.dumps(
            {"id": request_id, "method": method, "params": params or {}},
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(str(self.socket_path))
                connection.sendall(request)
                response_bytes = bytearray()
                while len(response_bytes) <= MAX_MESSAGE_BYTES:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    response_bytes.extend(chunk)
                    newline = response_bytes.find(b"\n")
                    if newline >= 0:
                        del response_bytes[newline:]
                        break
        except (FileNotFoundError, ConnectionRefusedError) as error:
            raise BridgeError(
                f"PoseWeaver is not reachable at {self.socket_path}. Launch it first."
            ) from error
        except OSError as error:
            raise BridgeError(f"PoseWeaver bridge connection failed: {error}") from error

        if len(response_bytes) > MAX_MESSAGE_BYTES:
            raise BridgeError("PoseWeaver bridge response exceeded 4 MiB")
        if not response_bytes:
            raise BridgeError("PoseWeaver closed the bridge connection without a response")
        try:
            response = json.loads(response_bytes)
        except json.JSONDecodeError as error:
            raise BridgeError("PoseWeaver returned invalid JSON") from error
        if response.get("id") not in (request_id, None):
            raise BridgeError("PoseWeaver returned a mismatched request ID")
        if not response.get("ok"):
            raise BridgeError(str(response.get("error", "unknown PoseWeaver error")))
        return response.get("result")

    def reachable(self) -> bool:
        try:
            self.call("app.status")
            return True
        except BridgeError:
            return False


class PoseWeaverController:
    def __init__(self, bridge: BridgeClient | None = None):
        self.bridge = bridge or BridgeClient()
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    def _release_child_handles(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            self.process.wait()
            self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def launch(self, arguments: dict[str, Any]) -> Any:
        try:
            status = self.bridge.call("app.status")
            return {**status, "already_running": True}
        except BridgeError:
            pass

        binary = Path(
            arguments.get("binary_path")
            or os.environ.get("POSEWEAVER_BINARY")
            or DEFAULT_BINARY
        ).expanduser().resolve()
        if not binary.is_file():
            raise BridgeError(
                f"PoseWeaver binary not found at {binary}. Run scripts/build.sh first."
            )
        if not os.access(binary, os.X_OK):
            raise BridgeError(f"PoseWeaver binary is not executable: {binary}")

        log_path = Path(arguments.get("log_path") or DEFAULT_LOG).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        environment["POSEWEAVER_AUTOMATION_SOCKET"] = str(self.bridge.socket_path)
        self.process = subprocess.Popen(
            [str(binary)],
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + float(arguments.get("startup_timeout_seconds", 20.0))
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self._log_tail(log_path)
                return_code = self.process.returncode
                self._release_child_handles()
                raise BridgeError(
                    f"PoseWeaver exited during startup with code {return_code}."
                    + (f" Log tail:\n{tail}" if tail else "")
                )
            try:
                status = self.bridge.call("app.status")
                return {
                    **status,
                    "already_running": False,
                    "binary_path": str(binary),
                    "log_path": str(log_path),
                }
            except BridgeError as error:
                last_error = str(error)
                time.sleep(0.1)
        raise BridgeError(f"PoseWeaver did not become ready before the timeout: {last_error}")

    def close(self, arguments: dict[str, Any]) -> Any:
        result = self.bridge.call("app.shutdown")
        timeout = float(arguments.get("timeout_seconds", 10.0))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.bridge.reachable():
                self._release_child_handles()
                return {**result, "closed": True}
            time.sleep(0.1)
        return {**result, "closed": False, "message": "shutdown was requested but the app is still reachable"}

    def status(self, _arguments: dict[str, Any]) -> Any:
        try:
            return self.bridge.call("app.status")
        except BridgeError as error:
            return {
                "running": False,
                "socket_path": str(self.bridge.socket_path),
                "message": str(error),
                "binary_path": str(DEFAULT_BINARY),
            }

    @staticmethod
    def _log_tail(path: Path, limit: int = 30) -> str:
        try:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])
        except OSError:
            return ""


def object_schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def tool(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "description": description, "inputSchema": schema}


TOOLS = [
    tool(
        "poseweaver_launch",
        "Launch PoseWeaver without controlling the desktop, or attach if it is already running.",
        object_schema(
            {
                "binary_path": {"type": "string", "description": "Optional absolute PoseWeaver binary path."},
                "log_path": {"type": "string", "description": "Optional application log path."},
                "startup_timeout_seconds": {"type": "number", "minimum": 1, "maximum": 120, "default": 20},
            }
        ),
    ),
    tool(
        "poseweaver_close",
        "Ask PoseWeaver to save its settings and close cleanly.",
        object_schema({"timeout_seconds": {"type": "number", "minimum": 0, "maximum": 60, "default": 10}}),
    ),
    tool("poseweaver_status", "Inspect PoseWeaver selections, pipelines, models, and simulation progress.", object_schema()),
    tool(
        "poseweaver_logs",
        "Read recent structured application console messages for diagnostics and operation results.",
        object_schema({"limit": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 100}}),
    ),
    tool(
        "poseweaver_assets",
        "List or refresh available poses, garments, and object meshes.",
        object_schema({"refresh": {"type": "boolean", "default": False}}),
    ),
    tool(
        "poseweaver_select",
        "Select PoseWeaver assets by exact listed name. Use null to clear an asset selection.",
        object_schema(
            {
                "gender": {"type": "string", "enum": ["male", "female", "neutral"]},
                "pose": {"type": ["string", "null"]},
                "garment": {"type": ["string", "null"]},
                "mesh": {"type": ["string", "null"]},
            }
        ),
    ),
    tool(
        "poseweaver_describe_parameters",
        "Return the writable PoseWeaver parameter registry with types, ranges, and descriptions.",
        object_schema(),
    ),
    tool(
        "poseweaver_get_parameters",
        "Read all writable PoseWeaver parameters, or only the requested parameter names.",
        object_schema({"names": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}}),
    ),
    tool(
        "poseweaver_set_parameters",
        "Atomically validate and set one or more PoseWeaver parameters. Describe parameters first when unsure.",
        object_schema({"values": {"type": "object", "additionalProperties": True}}, ["values"]),
    ),
    tool("poseweaver_list_layers", "List canvas layers, stable IDs, order, transforms, visibility, and selection.", object_schema()),
    tool(
        "poseweaver_manage_layer",
        "Import, select, rename, transform, hide, reorder, or remove a canvas layer.",
        object_schema(
            {
                "action": {"type": "string", "enum": ["import", "update", "remove", "reorder"]},
                "layer_id": {"type": "integer", "minimum": 1},
                "path": {"type": "string", "description": "PNG/JPEG path for import."},
                "name": {"type": "string"},
                "selected": {"type": "boolean"},
                "visible": {"type": "boolean"},
                "offset_x": {"type": "number"},
                "offset_y": {"type": "number"},
                "scale": {"type": "number", "exclusiveMinimum": 0},
                "index": {"type": "integer", "minimum": 0},
            },
            ["action"],
        ),
    ),
    tool(
        "poseweaver_export_image",
        "Export one layer's native RGBA canvas or the transformed composite of visible layers to PNG.",
        object_schema(
            {
                "path": {"type": "string"},
                "layer_id": {"type": "integer", "minimum": 1},
                "composite": {"type": "boolean", "default": False},
                "sketch_layers_only": {"type": "boolean", "default": False},
            },
            ["path"],
        ),
    ),
    tool("poseweaver_list_operations", "List supported application operations before starting one.", object_schema()),
    tool(
        "poseweaver_start_operation",
        "Start a typed PoseWeaver operation and return a job. Long operations continue asynchronously.",
        object_schema(
            {
                "name": {"type": "string"},
                "fine": {"type": "boolean", "description": "Use fine mesh refinement for render_cloth."},
                "play_on_finish": {"type": "boolean"},
                "continue_from_last_frame": {"type": "boolean"},
                "hide_rig_layer": {"type": "boolean"},
                "path": {"type": "string", "description": "Optional output path for scene_snapshot."},
            },
            ["name"],
        ),
    ),
    tool(
        "poseweaver_get_job",
        "Read a job's current state, stage, progress, timings, and output.",
        object_schema({"job_id": {"type": "integer", "minimum": 1}}, ["job_id"]),
    ),
    tool(
        "poseweaver_list_jobs",
        "List recent PoseWeaver jobs and their current states.",
        object_schema({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}}),
    ),
    tool(
        "poseweaver_cancel_job",
        "Safely cancel a running simulation job. Other pipeline kinds are intentionally not force-cancelled.",
        object_schema({"job_id": {"type": "integer", "minimum": 1}}, ["job_id"]),
    ),
]


class McpServer:
    def __init__(self, controller: PoseWeaverController | None = None):
        self.controller = controller or PoseWeaverController()
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "poseweaver_launch": self.controller.launch,
            "poseweaver_close": self.controller.close,
            "poseweaver_status": self.controller.status,
            "poseweaver_logs": lambda args: self.controller.bridge.call("app.logs", args),
            "poseweaver_assets": lambda args: self.controller.bridge.call("assets.refresh" if args.get("refresh") else "assets.list"),
            "poseweaver_select": lambda args: self.controller.bridge.call("selection.set", args),
            "poseweaver_describe_parameters": lambda _args: self.controller.bridge.call("parameters.describe"),
            "poseweaver_get_parameters": lambda args: self.controller.bridge.call("parameters.get", args),
            "poseweaver_set_parameters": lambda args: self.controller.bridge.call("parameters.set", args),
            "poseweaver_list_layers": lambda _args: self.controller.bridge.call("layers.list"),
            "poseweaver_manage_layer": self._manage_layer,
            "poseweaver_export_image": self._export_image,
            "poseweaver_list_operations": lambda _args: self.controller.bridge.call("operations.list"),
            "poseweaver_start_operation": lambda args: self.controller.bridge.call("operations.start", args),
            "poseweaver_get_job": lambda args: self.controller.bridge.call("jobs.get", args),
            "poseweaver_list_jobs": lambda args: self.controller.bridge.call("jobs.list", args),
            "poseweaver_cancel_job": lambda args: self.controller.bridge.call("operations.cancel", args),
        }

    def _manage_layer(self, arguments: dict[str, Any]) -> Any:
        action = arguments.get("action")
        methods = {
            "import": "layers.import",
            "update": "layers.set",
            "remove": "layers.remove",
            "reorder": "layers.reorder",
        }
        if action not in methods:
            raise BridgeError("unsupported layer action")
        params = {key: value for key, value in arguments.items() if key != "action"}
        return self.controller.bridge.call(methods[action], params)

    def _export_image(self, arguments: dict[str, Any]) -> Any:
        if arguments.get("composite"):
            return self.controller.bridge.call("layers.export_composite", arguments)
        if "layer_id" not in arguments:
            raise BridgeError("layer_id is required unless composite is true")
        return self.controller.bridge.call("layers.export", arguments)

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None
        try:
            if method == "initialize":
                requested_version = request.get("params", {}).get("protocolVersion", "2025-06-18")
                result = {
                    "protocolVersion": requested_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "poseweaver", "version": "1.0.0"},
                    "instructions": (
                        "Use PoseWeaver tools instead of mouse/keyboard control. Launch or check status first. "
                        "List assets before selecting names and describe parameters before setting unfamiliar keys. "
                        "Long operations return job IDs; poll poseweaver_get_job until completed or failed. "
                        "Use stable layer IDs, not list positions, for edits and exports."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if name not in self.handlers:
                    raise BridgeError(f"unknown tool: {name}")
                value = self.handlers[name](arguments)
                rendered = json.dumps(value, indent=2, ensure_ascii=False)
                result = {
                    "content": [{"type": "text", "text": rendered}],
                    "structuredContent": value if isinstance(value, dict) else {"result": value},
                    "isError": False,
                }
            else:
                return self._error(request_id, -32601, f"method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (BridgeError, OSError, ValueError, TypeError) as error:
            if method == "tools/call":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": str(error)}],
                        "isError": True,
                    },
                }
            return self._error(request_id, -32603, str(error))

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def run(self) -> None:
        for raw_line in sys.stdin.buffer:
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                response = self.dispatch(request)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                response = self._error(None, -32700, str(error))
            if response is not None:
                sys.stdout.write(json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-call",
        metavar="METHOD",
        help="Call the internal PoseWeaver bridge directly for diagnostics instead of running MCP.",
    )
    parser.add_argument("--params", default="{}", help="JSON object used with --bridge-call.")
    args = parser.parse_args()
    if args.bridge_call:
        try:
            params = json.loads(args.params)
            print(json.dumps(BridgeClient().call(args.bridge_call, params), indent=2))
            return 0
        except (BridgeError, json.JSONDecodeError) as error:
            print(str(error), file=sys.stderr)
            return 1
    McpServer().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
