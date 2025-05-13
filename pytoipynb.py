import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from pathlib import Path

# Input .py file name
py_filename = "compas_analysis.py"  # Change to your file name
ipynb_filename = Path(py_filename).with_suffix(".ipynb")

# Read the .py file and split into cells (based on '# In[' markers)
with open(py_filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

cells = []
current_cell = []

for line in lines:
    if line.strip().startswith("# In["):
        if current_cell:
            cells.append(new_code_cell("".join(current_cell)))
            current_cell = []
    else:
        current_cell.append(line)

if current_cell:
    cells.append(new_code_cell("".join(current_cell)))

# Create notebook object
nb = new_notebook(cells=cells, metadata={"language": "python"})

# Save as .ipynb
with open(ipynb_filename, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook saved as: {ipynb_filename}")
