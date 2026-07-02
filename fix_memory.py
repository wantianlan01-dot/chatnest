import re
path = r"C:\Users\34367\OneDrive\桌面\Codexx\chatnest\app\memory.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_func = r"async def memory_tool_permission\([\s\S]*?return PermissionResultAllow\(\)"
new_func = "async def memory_tool_permission(\n    tool_name: str,\n    tool_input: dict,\n    _context: object,\n):\n    return True  # claude-agent-sdk removed"
content = re.sub(old_func, new_func, content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
