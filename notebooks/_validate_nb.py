import json
with open('SDAIA-gpt-project2026/notebooks/05_model.ipynb') as f:
    nb = json.load(f)
print(f"Valid JSON, nbformat={nb['nbformat']}")
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = c['source']
    first_line = src[0][:70] if src else '(empty)'
    print(f"  [{i}] {c['cell_type']:10s} | {first_line}")
