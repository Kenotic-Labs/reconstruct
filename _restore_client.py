
import pathlib

content = pathlib.Path("sdk/client.py.restore").read_text(encoding="utf-8")
pathlib.Path("sdk/client.py").write_text(content, encoding="utf-8")
print("Restored")
