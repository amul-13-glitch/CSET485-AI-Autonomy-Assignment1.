# CSET485 – AI and Society · Assignment #1

**Does AI reduce human autonomy in decision making?** Stress-testing a credit-approval model on the German Credit dataset.

- Student: Amul Sharma · Enrollment S24BCAU0013 → German Credit dataset, `random_state = 13`
- Report: `Amul_Sharma_S24BCAU0013_Assignment1.pdf`

## Reproduce
```bash
pip install -r requirements.txt
python build_notebook.py                         # regenerates analysis.ipynb from analysis_src.py
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
python make_report.py Amul_Sharma_S24BCAU0013_Assignment1.pdf   # needs reportlab + Carlito fonts
```
`analysis.ipynb` (Kernel → Restart & Run All) loads `data/german.data`, fits OLS and logistic regression, runs the four stress tests
(Simpson's paradox scan, omitted-variable bias, adversarial subset, fairness sweep) plus the topic-lens extension, and writes every number
to `outputs/results.json` and every table/figure to `outputs/`. The PDF text reads its numbers from that file.

## Data
`data/german.data` is the original UCI Statlog German Credit symbolic file (1,000 rows, 20 attributes + class). Hofmann, H. (1994),
UCI Machine Learning Repository, https://doi.org/10.24432/C5NC77. Class 1 = Good → approve (1), class 2 = Bad → reject (0).
