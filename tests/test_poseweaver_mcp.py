import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "poseweaver_mcp.py"
SPEC = importlib.util.spec_from_file_location("poseweaver_mcp", SCRIPT)
assert SPEC and SPEC.loader
MCP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MCP)


class FakeBridge:
    def __init__(self):
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        return {"method": method, "params": params or {}}


class FakeController:
    def __init__(self):
        self.bridge = FakeBridge()

    def launch(self, arguments):
        return {"running": True, "arguments": arguments}

    def close(self, arguments):
        return {"closed": True, "arguments": arguments}

    def status(self, _arguments):
        return {"running": True}


class BridgeClientTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "bridge.sock"
            ready = threading.Event()

            def serve():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                    listener.bind(str(socket_path))
                    listener.listen(1)
                    ready.set()
                    connection, _ = listener.accept()
                    with connection:
                        request = json.loads(connection.makefile("rb").readline())
                        response = {"id": request["id"], "ok": True, "result": request}
                        connection.sendall(json.dumps(response).encode() + b"\n")

            thread = threading.Thread(target=serve)
            thread.start()
            self.assertTrue(ready.wait(2))
            result = MCP.BridgeClient(socket_path, timeout=2).call("app.status", {"detail": True})
            thread.join(2)
            self.assertEqual(result["method"], "app.status")
            self.assertEqual(result["params"], {"detail": True})


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.server = MCP.McpServer(self.controller)

    def test_initialize_negotiates_requested_version(self):
        response = self.server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        self.assertEqual(response["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_tool_catalog_has_unique_names(self):
        response = self.server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = response["result"]["tools"]
        names = [item["name"] for item in tools]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("poseweaver_start_operation", names)

    def test_layer_action_maps_to_bridge_method(self):
        response = self.server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "poseweaver_manage_layer",
                    "arguments": {"action": "update", "layer_id": 7, "visible": False},
                },
            }
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(
            self.controller.bridge.calls[-1],
            ("layers.set", {"layer_id": 7, "visible": False}),
        )

    def test_unknown_tool_returns_tool_error(self):
        response = self.server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "not_a_tool", "arguments": {}},
            }
        )
        self.assertTrue(response["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
