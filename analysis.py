"""Student performance analysis: data quality, group comparisons, statistics, charts and an Excel report.
Data: public 'Students Performance in Exams' practice dataset (1,000 students; math, reading and writing scores, 0-100).
The dataset is a public practice dataset and may be simulated, so findings describe this data only, not real students in general."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

NAVY, RED = "#1F3A5F", "#C0392B"
raw = pd.read_csv("data/students_performance.csv")
raw.columns = ["gender", "group", "parent_education", "lunch", "test_prep", "math", "reading", "writing"]
SCORES = ["math", "reading", "writing"]
EDU = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]

# ---------------------------------------------------------------- 1. data quality
iqr = lambda s: (s < s.quantile(.25) - 1.5 * (s.quantile(.75) - s.quantile(.25))) | (s > s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25)))
dq = pd.DataFrame([
    ("students (rows)", len(raw)), ("missing values", int(raw.isna().sum().sum())), ("duplicate rows", int(raw.duplicated().sum())),
    ("scores outside 0-100", int(((raw[SCORES] < 0) | (raw[SCORES] > 100)).sum().sum())),
    ("scores equal to 0", int((raw[SCORES] == 0).sum().sum())),
    ("text values with extra spaces", int(sum((raw[c] != raw[c].str.strip()).sum() for c in ["gender", "group", "parent_education", "lunch", "test_prep"]))),
    ("low outliers in math (IQR rule)", int(iqr(raw.math).sum())),
    ("low outliers in reading (IQR rule)", int(iqr(raw.reading).sum())),
    ("low outliers in writing (IQR rule)", int(iqr(raw.writing).sum())),
], columns=["check", "records"])
dq.to_csv("results/data_quality_report.csv", index=False)

df = raw.drop_duplicates().copy()
df["parent_education"] = pd.Categorical(df.parent_education, categories=EDU, ordered=True)
df["average"] = df[SCORES].mean(axis=1).round(2)
df["result_band"] = pd.cut(df.average, [-1, 49.99, 69.99, 100], labels=["Below 50", "50 - 69", "70 and above"])
df["at_risk"] = df.average < 50          # definition used in this project: average score below 50

# ---------------------------------------------------------------- 2. descriptive statistics
desc = df[SCORES + ["average"]].describe().T; desc["skewness"] = df[SCORES + ["average"]].skew(); desc.round(2).to_csv("results/descriptive_statistics.csv")
def summary(col):
    return df.groupby(col, observed=True).agg(students=("average", "size"), math=("math", "mean"), reading=("reading", "mean"), writing=("writing", "mean"),
                                              average=("average", "mean"), at_risk_pct=("at_risk", lambda s: 100 * s.mean())).round(2)
tabs = {"test_prep": summary("test_prep"), "parent_education": summary("parent_education"), "lunch": summary("lunch"), "gender": summary("gender"), "group": summary("group")}
for k, v in tabs.items(): v.to_csv(f"results/summary_by_{k}.csv")
pivot = df.pivot_table(index="parent_education", columns="test_prep", values="average", aggfunc="mean", observed=True).round(2)
pivot.to_csv("results/pivot_parent_education_x_test_prep.csv")

# ---------------------------------------------------------------- 3. statistical tests
def welch(col, a, b, score):
    x, y = df.loc[df[col] == a, score], df.loc[df[col] == b, score]
    t, p = stats.ttest_ind(x, y, equal_var=False); sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return {"comparison": f"{col}: {a} vs {b}", "score": score, "mean_1": x.mean(), "mean_2": y.mean(), "difference": x.mean() - y.mean(), "cohens_d": (x.mean() - y.mean()) / sd, "p_value": p}
rows = [welch("test_prep", "completed", "none", s) for s in SCORES + ["average"]] + [welch("lunch", "standard", "free/reduced", s) for s in SCORES + ["average"]] + \
       [welch("gender", "female", "male", s) for s in SCORES]
tt = pd.DataFrame(rows); tt.round(4).to_csv("results/group_comparisons.csv", index=False)
kw = stats.kruskal(*[g.average.values for _, g in df.groupby("parent_education", observed=True)])
rho_edu = stats.spearmanr(df.parent_education.cat.codes, df.average)
chi = stats.chi2_contingency(pd.crosstab(df.test_prep, df.lunch)); corr = df[SCORES].corr()
extra = pd.DataFrame([("Kruskal-Wallis: average score differs across parental education (H)", kw.statistic, kw.pvalue),
                      ("Spearman: parental education level vs average score (rho)", rho_edu.statistic, rho_edu.pvalue),
                      ("Chi-square: test prep course vs lunch type", chi.statistic, chi.pvalue),
                      ("Pearson correlation: reading vs writing", corr.loc["reading", "writing"], np.nan),
                      ("Pearson correlation: math vs reading", corr.loc["math", "reading"], np.nan),
                      ("Pearson correlation: math vs writing", corr.loc["math", "writing"], np.nan)], columns=["test", "value", "p_value"])
extra.round(4).to_csv("results/other_tests.csv", index=False)

# ---------------------------------------------------------------- 4. charts
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, s in zip(ax, SCORES): a.hist(df[s], bins=20, color=NAVY); a.set_title(f"{s.title()} score"); a.set_xlabel("Score")
plt.tight_layout(); plt.savefig("charts/score_distributions.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.2, 3.8)); w = .38; x = np.arange(3); t = tabs["test_prep"]
ax.bar(x - w/2, t.loc["completed", SCORES], w, label="Completed", color=NAVY); ax.bar(x + w/2, t.loc["none", SCORES], w, label="None", color=RED)
ax.set_xticks(x); ax.set_xticklabels([s.title() for s in SCORES]); ax.set_ylabel("Mean score"); ax.set_title("Test preparation course and mean scores"); ax.legend(); plt.tight_layout(); plt.savefig("charts/test_prep_effect.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.8)); e = tabs["parent_education"]; ax.barh(e.index.astype(str), e.average, color=NAVY); ax.set_xlabel("Mean average score"); ax.set_title("Mean average score by parental education"); plt.tight_layout(); plt.savefig("charts/parent_education.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(4.8, 3.8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=.5, vmax=1); ax.set_title("Correlation between subjects"); plt.tight_layout(); plt.savefig("charts/subject_correlation.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. Excel report
with pd.ExcelWriter("reports/student_performance_report.xlsx", engine="xlsxwriter") as xw:
    wb = xw.book; h = wb.add_format({"bold": True, "bg_color": NAVY, "font_color": "white", "border": 1}); num = wb.add_format({"num_format": "0.00"})
    lab = wb.add_format({"bold": True}); big = wb.add_format({"bold": True, "font_size": 13, "num_format": "0.00"})
    ws = wb.add_worksheet("Summary"); xw.sheets["Summary"] = ws
    for name, key in [("By Test Prep", "test_prep"), ("By Parent Education", "parent_education"), ("By Lunch", "lunch"), ("By Gender", "gender"), ("By Group", "group")]:
        t = tabs[key].reset_index(); t.iloc[:, 0] = t.iloc[:, 0].astype(str); t.to_excel(xw, sheet_name=name, index=False); w_ = xw.sheets[name]
        for i, c in enumerate(t.columns): w_.write(0, i, c, h)
        w_.set_column(0, 0, 22); w_.set_column(1, 6, 13, num)
    wpe = xw.sheets["By Parent Education"]; ch = wb.add_chart({"type": "bar"})
    ch.add_series({"name": "Average score", "categories": f"='By Parent Education'!$A$2:$A${len(EDU)+1}", "values": f"='By Parent Education'!$F$2:$F${len(EDU)+1}", "fill": {"color": NAVY}})
    ch.set_title({"name": "Mean average score by parental education"}); ch.set_legend({"none": True}); wpe.insert_chart("I2", ch)
    pivot.to_excel(xw, sheet_name="Pivot Education x Prep"); wp = xw.sheets["Pivot Education x Prep"]; wp.set_column(0, 3, 22, num)
    wp.conditional_format(1, 1, len(pivot), 2, {"type": "2_color_scale", "min_color": "#FFFFFF", "max_color": "#5B8DD6"})
    tt.round(4).to_excel(xw, sheet_name="Group Tests", index=False); wt = xw.sheets["Group Tests"]; wt.set_column(0, 0, 38); wt.set_column(1, 7, 14)
    for i, c in enumerate(tt.columns): wt.write(0, i, c, h)
    dq.to_excel(xw, sheet_name="Data Quality", index=False); wd = xw.sheets["Data Quality"]; wd.set_column(0, 0, 40)
    for i, c in enumerate(dq.columns): wd.write(0, i, c, h)
    ws.set_column(0, 0, 40); ws.set_column(1, 1, 16); ws.write(0, 0, "Student Performance Report", wb.add_format({"bold": True, "font_size": 16}))
    k = len(EDU) + 1
    ws.write(2, 0, "Students analysed", lab); ws.write_formula(2, 1, f"=SUM('By Parent Education'!B2:B{k})", big, int(len(df)))
    ws.write(3, 0, "Overall mean average score", lab); ws.write_formula(3, 1, f"=SUMPRODUCT('By Parent Education'!B2:B{k},'By Parent Education'!F2:F{k})/B3", big, float(df.average.mean()))
    ws.write(4, 0, "Students at risk (average below 50)", lab); ws.write_formula(4, 1, f"=SUMPRODUCT('By Parent Education'!B2:B{k},'By Parent Education'!G2:G{k})/100", big, float(df.at_risk.sum()))
    ws.write(5, 0, "Test prep: score gain (completed - none)", lab); ws.write_formula(5, 1, "='By Test Prep'!F2-'By Test Prep'!F3", big, float(t_gain := tabs['test_prep'].loc['completed','average'] - tabs['test_prep'].loc['none','average']))

# ---------------------------------------------------------------- 6. findings
tp = tt[(tt.comparison == "test_prep: completed vs none") & (tt.score == "average")].iloc[0]
lu = tt[(tt.comparison == "lunch: standard vs free/reduced") & (tt.score == "average")].iloc[0]
gm = tt[(tt.comparison == "gender: female vs male")].set_index("score")
e = tabs["parent_education"]
lines = [
 f"- {len(df)} students; no missing values or duplicates; one score of 0 and a few low outliers were kept and treated as valid scores.",
 f"- Average score overall: {df.average.mean():.1f}. {int(df.at_risk.sum())} students ({100*df.at_risk.mean():.1f}%) have an average below 50 ('at risk' in this project).",
 f"- Test preparation: students who completed the course scored {tp.difference:.1f} points higher on average ({tp.mean_1:.1f} vs {tp.mean_2:.1f}; Cohen's d = {tp.cohens_d:.2f}; p < 0.001).",
 f"- Lunch type: students with standard lunch scored {lu.difference:.1f} points higher on average than free/reduced lunch (d = {lu.cohens_d:.2f}; p < 0.001).",
 f"- Parental education: the mean average score is lowest for '{e.average.idxmin()}' ({e.average.min():.1f}) and highest for '{e.average.idxmax()}' ({e.average.max():.1f}); Kruskal-Wallis H = {kw.statistic:.1f}, p < 0.001; the overall trend is positive but modest (Spearman rho = {rho_edu.statistic:.2f}).",
 f"- Gender: girls scored {gm.loc['reading','difference']:.1f} points higher in reading and {gm.loc['writing','difference']:.1f} higher in writing; boys scored {-gm.loc['math','difference']:.1f} points higher in math.",
 f"- Reading and writing scores are very strongly correlated (r = {corr.loc['reading','writing']:.2f}); math is also strongly correlated with both (r = {corr.loc['math','reading']:.2f} and {corr.loc['math','writing']:.2f}).",
 "- These are associations, not proof of cause. For example, students who chose the prep course may differ in other ways.",
]
open("results/key_findings.md", "w").write("# Key findings (generated from the analysis output)\n\n" + "\n".join(lines) + "\n")
print(dq.to_string(index=False)); print(); print(tabs["test_prep"].to_string()); print(); print(e.to_string()); print(); print(tt.round(3).to_string(index=False)); print(extra.round(3).to_string(index=False)); print(); print("\n".join(lines))
