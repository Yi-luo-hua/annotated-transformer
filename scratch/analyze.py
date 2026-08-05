import json

# Analyze notebook
with open('AnnotatedTransformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Notebook Analysis ===")
print(f"Total cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(md_cells)}")
print()

for i, c in enumerate(code_cells):
    src = ''.join(c['source'])
    print(f"Code cell {i}: {repr(src[:150])}")
print()

# Analyze .py file
with open('the_annotated_transformer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

code_markers = [i for i, l in enumerate(lines) if l.strip() == '# %%']
print(f"=== .py Code Blocks: {len(code_markers)} ===")
for j, i in enumerate(code_markers):
    next_line = lines[i+1].strip()[:80] if i+1 < len(lines) else "EOF"
    print(f"  Block {j} at line {i+1}: {next_line}")
