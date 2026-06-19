"""
MCP Server Integration & Smoke Test.
Simulates a Model Context Protocol client by launching mcp_server.py as a subprocess,
sending JSON-RPC 2.0 requests over stdin/stdout, and verifying resources, tools, and prompts.
"""

import os
import sys
import json
import subprocess
import time

def read_json_line(process):
    """Helper to read one complete line of JSON from standard out."""
    line = process.stdout.readline()
    if not line:
        return None
    try:
        return json.loads(line.decode('utf-8'))
    except Exception as e:
        print(f"Failed to parse line: {line.decode('utf-8')}, error: {e}")
        return None

def write_json_line(process, data):
    """Helper to write JSON line to process standard input."""
    raw_data = json.dumps(data) + "\n"
    process.stdin.write(raw_data.encode('utf-8'))
    process.stdin.flush()

def run_smoke_test():
    print("[START] Starting MCP Server Smoke Test...")
    
    # Launch mcp_server.py in a subprocess with environmental configs
    env = os.environ.copy()
    env["DB_PASSWORD"] = "test-db-password-123"
    env["API_KEY"] = "sk-3ae47177f18e4ecf808440d6168c0d6f"
    env["API_PORT"] = "8080"
    
    proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(1.5) # Give the server a moment to start
    
    if proc.poll() is not None:
        _, err = proc.communicate()
        print(f"[ERROR] Server crashed on startup:\n{err.decode('utf-8')}")
        sys.exit(1)
        
    try:
        # 1. Handshake Initialize
        print("\n1. Performing Client Handshake...")
        init_req = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "AntigravitySmokeTester", "version": "1.0"}
            }
        }
        write_json_line(proc, init_req)
        
        # Read response
        init_res = read_json_line(proc)
        if not init_res or "error" in init_res:
            print(f"[ERROR] Handshake failed: {init_res}")
            proc.kill()
            sys.exit(1)
            
        print("[SUCCESS] Handshake successful. Protocol initialized.")
        
        # Send initialized notification
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        write_json_line(proc, init_notification)
        
        # 2. Query Resources
        print("\n2. Querying Registered Resources...")
        res_req = {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "id": 2
        }
        write_json_line(proc, res_req)
        
        res_res = read_json_line(proc)
        if not res_res or "result" not in res_res:
            print(f"[ERROR] Failed to list resources: {res_res}")
            proc.kill()
            sys.exit(1)
            
        resources = [r["uri"] for r in res_res["result"]["resources"]]
        print(f"[PASS] Registered Resources: {resources}")
        
        # Assert required resources
        expected_resources = ["patient://vitals", "patient://profile", "logs://anomalies"]
        for r in expected_resources:
            assert r in resources, f"Missing resource: {r}"
        print("[PASS] All required resources verified.")
        
        # 3. Query Tools
        print("\n3. Querying Registered Tools...")
        tools_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 3
        }
        write_json_line(proc, tools_req)
        
        tools_res = read_json_line(proc)
        if not tools_res or "result" not in tools_res:
            print(f"[ERROR] Failed to list tools: {tools_res}")
            proc.kill()
            sys.exit(1)
            
        tools = [t["name"] for t in tools_res["result"]["tools"]]
        print(f"[PASS] Registered Tools: {tools}")
        
        expected_tools = ["trigger_anomaly", "get_anomaly_logs", "generate_deepseek_insight", "generate_pdf_report"]
        for t in expected_tools:
            assert t in tools, f"Missing tool: {t}"
        print("[PASS] All required tools verified.")
        
        # 4. Call trigger_anomaly Tool
        print("\n4. Calling trigger_anomaly tool...")
        call_req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 4,
            "params": {
                "name": "trigger_anomaly",
                "arguments": {"state": "stable"}
            }
        }
        write_json_line(proc, call_req)
        
        call_res = read_json_line(proc)
        if not call_res or "result" not in call_res or "error" in call_res:
            print(f"[ERROR] Failed to call trigger_anomaly tool: {call_res}")
            proc.kill()
            sys.exit(1)
            
        text_out = call_res["result"]["content"][0]["text"]
        print(f"[PASS] Tool returned: '{text_out}'")
        
        # 5. Query Prompts
        print("\n5. Querying Registered Prompts...")
        prompts_req = {
            "jsonrpc": "2.0",
            "method": "prompts/list",
            "id": 5
        }
        write_json_line(proc, prompts_req)
        
        prompts_res = read_json_line(proc)
        if not prompts_res or "result" not in prompts_res:
            print(f"[ERROR] Failed to list prompts: {prompts_res}")
            proc.kill()
            sys.exit(1)
            
        prompts = [p["name"] for p in prompts_res["result"]["prompts"]]
        print(f"[PASS] Registered Prompts: {prompts}")
        assert "analyze_arrhythmia" in prompts
        print("[PASS] Required prompt verified.")
        
        print("\n[SUCCESS] ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
        
    finally:
        # Shutdown server gracefully
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()
            
if __name__ == "__main__":
    run_smoke_test()
