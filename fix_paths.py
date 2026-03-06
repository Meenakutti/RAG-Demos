# fix_paths.py
from pathlib import Path
p = Path("modules/2_chunking/demo.py")
s = p.read_text(encoding="utf-8")
s = s.replace('../../data/synthetic_tickets.json', 'DATA_FILE')
if 'DATA_FILE' in s and 'from pathlib import Path' not in s:
    s = 'from pathlib import Path\nDATA_FILE = Path(__file__).resolve().parents[2] / "data" / "synthetic_tickets.json"\n\n' + s
p.write_text(s, encoding="utf-8")
print("Patched", p)