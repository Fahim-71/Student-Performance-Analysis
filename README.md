# Student Performance Analysis (Education) with pandas, Statistics and Excel

Data analysis project on exam scores of 1,000 students (math, reading and writing, scored 0-100) with background information
(gender, parental education, lunch type, test preparation course). It looks for factors linked to higher or lower scores.

**Dataset:** the public "Students Performance in Exams" practice dataset (copy in `data/students_performance.csv`).
It is a practice dataset and may be simulated, so the findings describe this data only, not real students in general.

## What the analysis does
1. **Data quality checks:** missing values, duplicates, scores outside 0-100, zeros, text with extra spaces, IQR outliers
2. **Feature creation with pandas:** average score, result bands, and an "at risk" flag (average below 50)
3. **Group summaries and pivot table:** by test preparation, parental education, lunch type, gender and group; parental education x test prep pivot
4. **Statistics:** Welch t-tests with Cohen's d effect sizes, Kruskal-Wallis test, Spearman correlation, chi-square test, subject correlation matrix
5. **Charts and Excel report** with live formulas, a native chart and a colour-scale pivot table

## Key findings
Full list in `results/key_findings.md` (generated from the real output):
- Students who completed the test preparation course scored 7.6 points higher on average (Cohen's d = 0.56)
- Students with standard lunch scored 8.6 points higher on average than those with free or reduced lunch
- Girls scored higher in reading and writing; boys scored higher in math
- 103 students (10.3%) have an average score below 50
- Results are associations, not proof of cause

## Files
```
analysis.py         the full analysis (run this)
data/               dataset
results/            CSV tables and key_findings.md
charts/             PNG charts
reports/student_performance_report.xlsx   Excel report
```

## Run
```bash
pip install -r requirements.txt
python analysis.py
```
