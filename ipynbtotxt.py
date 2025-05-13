import nbformat
from nbconvert import PythonExporter
import sys
from pathlib import Path

# === SETTINGS ===
notebook_filename = "compas_analysis.ipynb"  # Change this to your notebook file name

# === LOAD AND CONVERT ===
with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
(body, resources) = exporter.from_notebook_node(nb)

# === WRITE TO .py FILE ===
py_filename = Path(notebook_filename).with_suffix('.py')
with open(py_filename, 'w', encoding='utf-8') as f:
    f.write(f"# Converted from notebook: {notebook_filename}\n\n")
    f.write(body)

print(f"Saved as {py_filename}")
