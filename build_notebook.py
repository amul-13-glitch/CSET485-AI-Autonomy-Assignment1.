"""Builds analysis.ipynb from analysis_src.py ('# %%' = code cell, '# %% [markdown]' = markdown cell)."""
import nbformat as nbf, re
src = open("analysis_src.py").read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    first, _, body = chunk.partition("\n")
    if "[markdown]" in first:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        cells.append(nbf.v4.new_code_cell(body.strip()))
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "analysis.ipynb")
