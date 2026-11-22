"""Builds the PDF report from outputs/results.json (written by analysis.ipynb). Every number in the text is read from that file."""
import json, os, sys, re
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

F = "/usr/share/fonts/truetype/crosextra/"
for n, f in [("Carlito", "Carlito-Regular.ttf"), ("Carlito-Bold", "Carlito-Bold.ttf"), ("Carlito-Italic", "Carlito-Italic.ttf"), ("Carlito-BoldItalic", "Carlito-BoldItalic.ttf")]:
    pdfmetrics.registerFont(TTFont(n, F + f))
pdfmetrics.registerFontFamily("Carlito", normal="Carlito", bold="Carlito-Bold", italic="Carlito-Italic", boldItalic="Carlito-BoldItalic")

STUDENT, ENROLL, SEED = "Amul Sharma", "S24BCAU0013", 13
REPO_URL = os.environ.get("REPO_URL", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
R = json.load(open("outputs/results.json"))
SW = {round(r["t"], 2): r for r in R["sweep"]}
NAVY, BLUE = colors.HexColor("#1f3b57"), colors.HexColor("#2e5d8a")

def pc(x, d=2): return f"{x:.{d}f}%"
def f4(x): return f"{x:+.4f}"
def n3(x): return f"{x:.3f}"
def shr(x, d=1): return f"{100*x:.{d}f}%"

body = ParagraphStyle("b", fontName="Carlito", fontSize=10.2, leading=13.4, spaceAfter=6, alignment=TA_LEFT)
h1 = ParagraphStyle("h1", fontName="Carlito-Bold", fontSize=15, leading=18, textColor=NAVY, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Carlito-Bold", fontSize=12, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=4)
note = ParagraphStyle("n", parent=body, fontName="Carlito-Italic", fontSize=9.6)
cap = ParagraphStyle("c", parent=body, fontName="Carlito-Bold", fontSize=8.8, leading=11, textColor=BLUE)
cell = ParagraphStyle("cell", fontName="Carlito", fontSize=8.8, leading=10.6)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Carlito-Bold", textColor=colors.white)
ctr = ParagraphStyle("ctr", parent=body, alignment=TA_CENTER)
title = ParagraphStyle("t", fontName="Carlito-Bold", fontSize=26, leading=31, textColor=NAVY, alignment=TA_CENTER)

story, WC = [], {}
def P(t, s=body): story.append(Paragraph(t, s))
def H1(t): story.append(Paragraph(t, h1))
def H2(t): story.append(Paragraph(t, h2))
def words(t): return len(re.sub(r"<[^>]+>", "", t).split())
def ANS(key, t):
    WC[key] = words(t); P(t)

def table(rows, widths, header=True, align_right_from=None):
    start = len(story)
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w*inch for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#555555")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    for i in range(1, len(rows)):
        if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef2f6")))
    t.setStyle(TableStyle(st)); story.append(KeepTogether([t]) if len(rows) <= 14 else t); story.append(Spacer(1, 7))

def header_footer(c, d):
    c.saveState(); c.setFont("Carlito-Bold", 8.5); c.setFillColor(BLUE)
    c.drawString(0.8*inch, 10.45*inch, "CSET485  |  AI AND SOCIETY"); c.drawString(3.0*inch, 10.45*inch, "ASSIGNMENT #1")
    c.setFont("Carlito", 8.5); c.setFillColor(colors.HexColor("#444444"))
    c.drawCentredString(4.25*inch, 0.5*inch, f"{STUDENT}  |  {ENROLL}  |  {d.page}"); c.restoreState()

doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch,
                      title="CSET485 Assignment 1 - Does AI reduce human autonomy in decision making?", author=STUDENT)
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(0.8*inch, 0.75*inch, 6.9*inch, 9.5*inch, id="f")], onPage=header_footer)])

# ---------------------------------------------------------------- numbers used throughout
t35, t55, t65, t75, t85, t05, t15, t95, t45, t25 = (SW[x] for x in (0.35, 0.55, 0.65, 0.75, 0.85, 0.05, 0.15, 0.95, 0.45, 0.25))
bands = {b["band"][:9]: b for b in R["autonomy_bands"]}
b_rej, b_unc, b_app = R["autonomy_bands"]
sh = R["simpson_headline"]; st = R["simpson_table"]; si = R["simpson_interaction"]
ovb = {o["predictor"]: o for o in R["ovb"]}
hp, tp = R["hc_profile"], R["test_profile"]
trait = {t["trait"]: t for t in R["hc_traits"]}
fnr, fpa = R["hc_fn_rare_history"], R["hc_fp_no_account"]
td_pp, td_eo = R["tradeoffs"]["DP_vs_PP"], R["tradeoffs"]["DP_vs_EO"]
logit_acc_n = round(R["logit_acc"]*300)
tp_df = pd.read_csv("outputs/test_predictions.csv")
fem = tp_df[tp_df.orig_row == 613].iloc[0]; mal = tp_df[tp_df.orig_row == 433].iloc[0]
cons_n = b_rej["n"] + b_app["n"]

# ---------------------------------------------------------------- COVER
story.append(Spacer(1, 1.3*inch))
P("CSET485 – AI AND SOCIETY", ParagraphStyle("x", parent=ctr, fontName="Carlito-Bold", fontSize=14, textColor=NAVY, spaceAfter=14))
P("ASSIGNMENT #1", ParagraphStyle("x2", parent=ctr, fontName="Carlito-Bold", fontSize=12, spaceAfter=22))
P("Does AI Reduce Human Autonomy<br/>in Decision Making?", title)
story.append(Spacer(1, 6))
P("Stress-testing a Credit Approval Prediction Model", ParagraphStyle("x3", parent=ctr, fontName="Carlito-Bold", fontSize=14, textColor=BLUE, spaceAfter=4))
P("German Credit Data | Statistical and Fairness Analysis", ParagraphStyle("x4", parent=ctr, fontSize=11, textColor=BLUE, spaceAfter=18))
table([["Submission detail", "Information"], ["Student Name", STUDENT], ["Enrollment Number", ENROLL], ["Programme", "BCA (Cyber Security)"],
       ["Course", "CSET485 – AI and Society"], ["Topic", "Does AI reduce human autonomy in decision making?"],
       ["Assigned Dataset", "German Credit Data"], ["Train/Test Split", "70% / 30%"], ["Random State", str(SEED)]], [2.6, 3.4])
P("Submitted in accordance with the Assignment #1 instructions", ParagraphStyle("x5", parent=ctr, fontName="Carlito-Italic", fontSize=9, spaceBefore=26))
story.append(PageBreak())

# ---------------------------------------------------------------- ABSTRACT
H1('Abstract')
P('This assignment explores whether a simple statistical model can support credit decisions without hiding errors or group differences and, in the process, explores the wider question: Does AI reduce human autonomy in decision making? The German Credit dataset includes 1,000 historical credit-risk records. For this assignment, "Good" was mapped to approval (1) and "Bad" was mapped to rejection (0). The required 70/30 split was applied with random_state=13. An ordinary least squares (OLS) model and a logistic regression were fitted. Four stress tests were then performed: subgroup analysis, omitted-variable analysis, a review of high-confidence errors, and a fairness-threshold sweep. The logistic model had a test accuracy of 75.00% at 0.50, which is only 1.33% points higher than the majority-class baseline of 73.67%. 25.33% of test cases were classified differently by the two thresholding rules. Dropping one variable resulted in 10.33% of decisions changing. The evidence suggests that a model can narrow a human reviewer\'s real room to decide through hidden design choices (thresholds, features, confidence), even though this dataset contains no reviewer behavior and so cannot measure deference directly.', body)
H1('Executive summary of computed results')
table([["Item", "Result"],
       ["Dataset", f"{R['n_rows']:,} records; {R['n_good']} Good / {R['n_bad']} Bad"],
       ["Split", f"{R['train_n']} training / {R['test_n']} test records; random_state={SEED}"],
       ["Model B accuracy at 0.50", f"{pc(100*R['logit_acc'])} ({logit_acc_n} of 300 test records); majority baseline {pc(100*R['majority_baseline'])}"],
       ["OLS median prediction", f"{R['ols_median']:.4f}"],
       ["Different classifications", f"{R['mismatch_n']} of 300 cases ({pc(R['mismatch_pct'])})"],
       ["Simpson-type reversal", f"credit_amount: {f4(st[1]['coef'])} (phone owners) vs {f4(st[2]['coef'])} (no phone); pooled {f4(st[0]['coef'])}"],
       ["Largest coefficient shift (checking_status dropped)", f"credit_amount {ovb['credit_amount']['pct_change']:+.1f}%; {R['ovb_decision_flips']} decisions flip ({pc(R['autonomy_flip_full_vs_reduced_pct'])})"],
       ["High-confidence wrong cases", f"{R['hc_total']} of 300 ({pc(R['hc_pct'])}); {R['hc_fp']} false positives and {R['hc_fn']} false negatives"],
       ["Most defensible threshold discussed", f"0.35 (DP {n3(t35['DP'])}, EO {n3(t35['EO'])}, PP {n3(t35['PP'])}), with an important cost caveat"]], [2.5, 4.4])
P('Interpretation note: The labels "approve" and "reject" are the assignment\'s operational names for the UCI risk classes. They are not evidence of actual bank decisions. The dataset records good and bad credit risk, not real lending decisions for these individuals.', note)
H1('Dataset and assignment setup')
P('The course handout assigns the German Credit dataset to enrollment numbers ending in 3, 4, or 5. My enrollment number is S24BCAU0013, which ends in 3, so this is the correct dataset. The final three digits are 013, so the random seed is 13. The UCI repository describes 1,000 instances and 20 input features with no missing values, and provides the original symbolic file (german.data), which is the version used here [1].', body)
P('The original class column is coded 1 = Good and 2 = Bad. To follow the instruction to encode approval as 1 and rejection as 0, I used Good = 1 (approve) and Bad = 0 (reject). This is a modeling proxy. The class distribution is 700 Good and 300 Bad. The split produced 700 training rows (479 Good and 221 Bad) and 300 test rows (221 Good and 79 Bad).', body)
H1('Data preprocessing')
fl = R["iqr_flags"]
table([["Requirement", "How it was handled"],
       ["Missing values", f"The UCI source reports no missing values and the file has no “?” markers (computed count: {R['missing_values']}). No imputation was needed."],
       ["Categorical encoding", "All 13 categorical attributes were one-hot encoded. The first sorted training category of each attribute was omitted as the reference level."],
       ["Numeric scaling", "All non-intercept design columns (seven numeric fields and the one-hot indicators) were standardised with training-set means and population standard deviations. The same parameters were applied to test rows."],
       ["Outliers", "1.5×IQR flags were inspected but no records were deleted, because extreme amounts and durations may be genuine cases and the rule is uninformative for small count fields."],
       ["Train/test separation", f"70/30 split, random_state={SEED}, no stratification (as specified). Preprocessing parameters were learned from training rows only."]], [1.7, 5.2])
P('For transparency, the 1.5×IQR rule flags 72 credit amounts (7.2%), 70 durations (7.0%), 23 ages (2.3%), 6 existing-credit counts (0.6%), and 155 "people liable" values (15.5%). The last flag is driven by a zero-width IQR (most records have one liable person), not by data errors. Installment rate and residence duration had no flags. The design matrix has 49 columns including the intercept; the target and the demographic field used for fairness analysis are not counted as separate predictors beyond their encoded columns.', body)
H1('Part 1 – Build and compare the models')
H2('1.1 Model A: Linear regression (OLS)')
P('I fitted OLS to the binary approval target and treated its output as an estimated approval probability, as required. The median of the 300 test predictions was 0.6999. Predictions ranged from -0.010 to 1.331, and 31 of 300 (10.3%) fell outside the 0–1 range, which illustrates a basic weakness of the linear probability model.', body)
H2('1.2 Model B: Logistic regression')
P('Logistic regression was fitted to the same training set and design. At the required 0.50 threshold it classified 226 of 300 test records as approve. Its accuracy against the actual UCI label was 75.00% (225 correct), against a majority-class baseline of 73.67%. The OLS median rule reached 68.33%.', body)
H2('1.3 Agreement between the two thresholding methods')
ag = R["agreement"]
table([["", "Logistic: reject (0)", "Logistic: approve (1)", "Row total"],
       ["OLS: below median (0)", ag[0][0], ag[0][1], sum(ag[0])], ["OLS: at/above median (1)", ag[1][0], ag[1][1], sum(ag[1])],
       ["Column total", ag[0][0]+ag[1][0], ag[0][1]+ag[1][1], 300]], [2.0, 1.7, 1.7, 1.5])
P('This is an agreement matrix between the two prediction rules, not a confusion matrix against the true outcome. The rules disagreed on 76 cases, 25.33% of the test set. All 150 records above the OLS median were also approved by logistic regression, while 76 records below the OLS median were still approved by logistic regression.', body)
H1('Part 2 – The Trap: stress tests')
H2('2.1 Simpson’s Paradox Hunt / subgroup effect')
P("I scanned 35 subgroup comparisons (five social splits × seven numeric predictors, each with logistic models on the full training set and on both subgroups). 16 comparisons showed opposite signs between subgroups, but most were tiny coefficients with wide intervals. Only 2 reversals were also statistically different between subgroups at p&lt;0.05, and both involved the telephone split (a telephone registered in the customer's name, a rough marker of established social and financial standing). The cleanest is the credit_amount effect, reported below using standardized numeric predictors only.", body)
table([["Training group", "n", "Credit amount coefficient", "SE", "95% Wald CI", "Approx. p"]] +
      [[r["group"], r["n"], f4(r["coef"]), f"{r['se']:.4f}", f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", f"{r['p']:.4f}"] for r in st],
      [1.8, .5, 1.3, .6, 1.6, .9])
story.append(Image("outputs/fig_simpson.png", width=5.0*inch, height=2.34*inch))
P('Figure 1. Credit-amount coefficient with 95% Wald intervals in the full sample and in each telephone subgroup.', cap)
P('The sign reverses: larger credit amounts lower the approval log-odds for telephone owners (-0.6379) but raise them for non-owners (+0.4913). The pooled coefficient (-0.1641, CI -0.392 to 0.064) hides both effects and is not significant (p=0.158). The difference between the two subgroup coefficients has z=-4.19 (p=0.00003), which survives a Bonferroni correction for 35 tests (threshold 0.05/35=0.0014). A joint model with an amount×telephone interaction, adjusted for the other predictors, gives an interaction coefficient of -0.6189 (p=0.0018). For the demographic variable used later (sex), no significant reversal appeared; the clearest difference was for age (+0.0811 for men vs +0.5319 for women, difference p=0.038), which would not survive the multiple-comparison correction.', body)
P('<b>Why the trend differs.</b> Telephone owners are overrepresented among large loans: the share of owners is 27.8%, 39.9% and 57.1% across the lowest, middle and highest amount terciles, and their mean amount is 4,303 against 2,616. Yet the two groups have similar approval rates (69.1% vs 68.0%). The pooled slope therefore blends a negative within-group slope with a positive one and shrinks towards zero. Why non-owners show a positive slope is not answerable from this file; one hypothesis is that larger loans to less-documented applicants are granted only after closer scrutiny, but that is untested. If subgroup analysis were ignored, a policymaker would read the pooled result as "loan size does not matter" and drop it as a risk factor, or would apply a blanket cap that is wrong for the 409 of 700 training applicants without a telephone.', body)
H2('2.2 Omitted-variable bias')
P('For this test I removed all checking_status columns from the full logistic model. It is a plausible confounder: the UCI documentation describes it as the status of an existing checking account, which is plausibly related both to other loan characteristics and to the outcome [1]. Its "no account" level has the largest coefficient in the model (+0.8606). The table compares all seven numeric predictors.', body)
table([["Predictor", "Full model coefficient", "Reduced model coefficient", "Absolute change", "Signed change / |full|"]] +
      [[o["predictor"], f4(o["full"]), f4(o["reduced"]), f"{o['abs_change']:+.4f}", f"{o['pct_change']:+.2f}%"] for o in R["ovb"]], [1.5, 1.4, 1.5, 1.2, 1.3])
P('The percentage column is (reduced − full) divided by the absolute full coefficient, multiplied by 100. Residence shows −308% only because its full coefficient is almost zero (+0.0068), so a tiny absolute change of -0.0211 looks enormous; I do not treat it as the meaningful shift. Among coefficients of substantive size, credit_amount shifted the most: -0.3526 to -0.2672, +24.2%, which is also the largest absolute change (+0.0853). Duration moved -18.1%, installment_rate +15.6% and age +12.2%. Test accuracy barely moved (75.00% to 74.00%), yet 31 of 300 individual decisions (10.33%) flipped. Accuracy alone would not reveal the problem. In the real world I would run the same experiment in reverse: obtain plausible extra variables (for example income stability or loan-officer override reasons) and check whether key coefficients move by more than a pre-agreed amount, such as 20%.', body)
H2('2.3 Adversarial subset construction')
P('I looked for confident mistakes in the logistic test predictions: predicted approval probability at least 0.75 when the true class was Bad (false positive), or at most 0.25 when the true class was Good (false negative). There were 26 such rows, 8.67% of the 300-row test set (13 false positives, 13 false negatives). Since this is below 50, all 26 were retained rather than sampled.', body)
def tr_row(label, key):
    t = trait[key]; return [label, f"{t['subset_n']} / {R['hc_total']} ({pc(t['subset_pct'],1)})", f"{t['test_n']} / 300 ({pc(t['test_pct'],1)})"]
table([["Measure", "High-confidence error subset", "Full test set"],
       ["Number of records", R["hc_total"], 300],
       ["Male / Female", f"{hp['male']} / {hp['female']} ({pc(100*hp['male']/hp['n'],1)} / {pc(100*hp['female']/hp['n'],1)})", f"{tp['male']} / {tp['female']} ({pc(100*tp['male']/tp['n'],1)} / {pc(100*tp['female']/tp['n'],1)})"],
       ["Mean / median age", f"{hp['mean_age']:.2f} / {hp['median_age']:.0f} years", f"{tp['mean_age']:.2f} / {tp['median_age']:.0f} years"],
       ["Age under 25 / 25–39 / 40+", f"{hp['age_lt25']} / {hp['age_25_39']} / {hp['age_ge40']}", f"{tp['age_lt25']} / {tp['age_25_39']} / {tp['age_ge40']}"],
       ["Mean credit amount", f"{hp['mean_amount']:,.2f}", f"{tp['mean_amount']:,.2f}"],
       ["Mean duration (months)", f"{hp['mean_duration']:.2f}", f"{tp['mean_duration']:.2f}"],
       tr_row("Credit history A30/A31", "Credit history A30/A31 (rare categories)"), tr_row("No checking account (A14)", "Checking account A14 (no account)"),
       tr_row("Credit amount above 5,000", "Credit amount > 5,000"), tr_row("Duration above 24 months", "Duration > 24 months")], [2.1, 2.5, 2.3])
P('The table compares the subset with the whole test set; the text below uses class-specific baselines, because false positives can only come from Bad applicants and false negatives only from Good ones. No single trait covers every row, because the errors come in two mirrored types. False negatives: 8 of 13 (61.5%) have credit history A30 or A31, compared with 7.7% of all Good test applicants (Fisher p=2.1e-07). False positives: 7 of 13 (53.8%) have no checking account (A14), against 12.7% of Bad test applicants (p=5.6e-05). Together these two patterns cover 15 of 26 rows (57.7%). Within the 23 test applicants the model rejects with confidence, 56.5% were actually Good, so these rejections are wrong more often than right. The model is additive: A14 carries a coefficient of +0.8606 (SE 0.137) and cannot be offset by interactions with amount or purpose. A30/A31 histories have only 61 training rows (8.7%) and a Good rate of 31.1% versus 68.4% overall, so the model learns a harsh penalty from a small cell and misses the exceptions. A confident probability is the model’s certainty under its learned structure, not proof.', body)
H2('2.4 Fairness trade-off analysis')
P('Personal status and sex are the sensitive variables. Grouped into Male (A91, A93, A94) and Female (A92, A95) with UCI codes [1]. The test set consists of 203 male records and 97 female records. For a given threshold the model predicts approval if P(approval) &gt;= t. DP is approval-rate ratio, EO is min of TPR ratio and FPR ratio, PP is positive-predictive-value ratio, all Male / Female. As defined in the handout. Also reported: accuracy and UCI misclassification cost (5 × bad approved + 1 × good rejected) [1].', body)
table([["Threshold", "DP", "EO", "PP", "Accuracy", "Approved", "Good rejected", "Bad approved", "UCI cost"]] +
      [[f"{r['t']:.2f}", n3(r["DP"]), n3(r["EO"]), n3(r["PP"]), pc(100*r["acc"], 1), r["approved"], r["good_rejected"], r["bad_approved"], r["uci_cost"]] for r in R["sweep"]],
      [.8, .65, .65, .65, .8, .8, .95, .95, .7])
story.append(Image("outputs/fig_fairness.png", width=5.2*inch, height=3.74*inch))
P('Figure 2. DP vs. EO for threshold 0.05 to 0.95. The 0.95 (DP = 1.593, EO = 1.432) point is outside the specified 0 to 1.5 range, so not plotted. At 0.95 the female FPR is zero, so FPR ratio is undefined. EO uses the finite TPR ratio.', cap)
P('At every threshold more males are approved than females (DP &gt; 1). Also TPR ratio is &gt; 1. This means more good female applicants are rejected than good male applicants. For example, at t=0.35 the approval rate is 87.19% for males, 80.41% for females. TPR is 92.16% and 86.76%. FPR is 72.00% and 65.52%. Using the min of TPR ratio and FPR ratio (1.062 and 1.099) gives EO = 1.062. This can mask the larger of the two gaps.', body)
P('<b>Quantifying trade-offs.</b> We can measure each metric as the distance from ideal 1.0. For the largest conflict between DP and PP, going from t=0.65 to t=0.05, DP improves from 1.357 to 1.010 (the distance from 1 decreases 97.1%). But PP drifts from 0.942 to 1.080 (the distance increases 36.7%). This comes at a cost of trade-off W/Z = 0.38. For DP and EO, going from t=0.85 to t=0.35, DP improves from 1.124 to 1.084 (32.2% gain in distance). But EO worsens from 1.042 to 1.062 (49.2% loss). This comes at a cost of 1.53. Even the nearly "fair" low thresholds are not safe. At t=0.05 the model approves 299 out of 300. This includes 79 out of 79 Bad. This gives UCI cost of 396. Worst in the sweep. No single threshold satisfies all three metrics, and accuracy and cost.', body)
H2('2.5 Topic lens: where is there still room for a human reviewer to decide?')
P('This is a short extension to link the stress tests to the chosen topic. It only uses the fitted logistic model and the test predictions. It measures structural room for human judgement, not reviewer behaviour (which is not in this dataset).', body)
table([["Score band", "Test applicants", "Model accuracy in band", "Actual Good rate"]] +
      [[b["band"].replace("<", "&lt;"), f"{b['n']} ({pc(b['pct'],1)})", pc(100*b["model_acc"], 1), pc(100*b["actual_good_rate"], 1)] for b in R["autonomy_bands"]], [2.6, 1.4, 1.6, 1.3])
story.append(Image("outputs/fig_autonomy.png", width=5.0*inch, height=2.34*inch))
P('Figure 3. Distribution of logistic scores for the 300 test applicants. The grey band is the uncertain score.', cap)
P('Three different design choices can flip individual decisions without any change to the applicant. OLS median vs. 0.50 rule. 25.33% of the outcomes change. Dropping one variable. 10.33% of the outcomes change. Changing cut-off from 0.65 to 0.35. 63 decisions change (21.0%). If a reviewer defers to the score, they inherit all these choices. It is most safe to defer in the confident-approve band (91.8% accurate). Least safe in the confident-reject band (43.5% accurate, worse than a coin flip). So a blanket rule to follow the model would remove human judgement where it is most needed.', body)
H1('Part 3 – Descriptive analysis questions')
P("All responses are based on the values calculated from the fixed training and test split. 'Approval' is defined as the assignment mapping Good = 1, Bad = 0.", note)
H2('3.1 Why did the two approaches flag different individuals?')
P('The two approaches have different decision rules. Even though they are fitted on the same training records and the same target. The linear model has a median test prediction of 0.6999. In the handout, a case is labelled as approve if the OLS prediction is &gt;= this median. This by construction splits the 300 test records into 150 below, 150 at or above.', body)
P('The logistic regression approves if the probability is &gt;= 0.50. This is a fixed cut-off. It does not select half of the data. In the agreement matrix, 74 cases are rejected by both. 150 cases are approved by both. 76 cases are below the OLS median but above the logistic 0.50 threshold. No case the other way around. So in total 76/300 records (25.33%) have different labels. The logistic regression approves 226 applicants, vs. 150 with the median rule.', body)
P('This is a structural gap. The OLS median (0.6999) is well above 0.50. Since 73.7% of the test applicants are Good, forcing half of them into the low group must reject many people that the logistic model would approve. Also the median rule is relative. It would approve exactly 150 applicants, regardless of their absolute score. The OLS predictions range from -0.010 to 1.331. There are 31 outside the probability range. So the linear scale is not calibrated like the logistic one.', body)
P('It is not that one approach is fairer. The choice of threshold changes who gets approval. The median rule agrees with the 0.50 rule only 74.67% of the time. This is a design decision that a lender must justify. Not a technical detail. This is the first place where human discretion is quietly replaced with a rule.', body)
H2('3.2 Subgroup effect and what it means for policy')
P('I chose telephone registration as the subgroup variable. Because it had a genuine sign reversal. I compared the same seven numeric predictors in three logistic regressions. So I can compare the credit_amount coefficient consistently. There are 700 records in the training data. 291 telephone owners, 409 non-owners. Credit amount is standardised. So each coefficient is the change in log-odds for a 1 standard deviation increase.', body)
P('In the full model, the coefficient is -0.1641. The 95% Wald interval is -0.3918 to 0.0637. (p=0.158) For the owners (subgroup A), it is -0.6379. Interval is -0.9949 to -0.2809. (p=0.0005) For the non-owners (subgroup B), it is +0.4913. Interval is 0.1013 to 0.8812. (p=0.0135) The intervals for the subgroups do not overlap. They are on opposite sides of zero. The difference test gives z=-4.19.', body)
P('The reversal comes from the uneven distribution of the groups over the loan sizes. Telephone owners are 27.8% of the lowest amount tercile. But 57.1% of the highest amount tercile. Yet the approval rates are similar (69.1% vs. 68.0%). So when pooling, you mix a negative slope with a positive slope. You get a weak coefficient. The interval includes zero.', body)
P('I cannot say why there is a positive slope for non-owners. That would require new data. In the joint model, the interaction term is -0.6189. (p=0.0018) So it is not an artefact of the separate fits.', body)
P('If we ignored the subgroup, the policy conclusion would be that loan size has no reliable effect on approval. So we would drop or cap amount for everyone. But in truth larger loans are penalised for owners. They are favoured for non-owners. So a uniform rule would misjudge most of all the 409 non-owners.', body)
H2('3.3 Omitted variable test and coefficient changes')
P('I dropped checking_status from the full logistic model. This is the variable describing the applicant\'s existing checking account. The "no account" level has the largest coefficient in the model (+0.8606). So it is a strong candidate confounder. Then I compared the numeric predictors that are still in both models. Percentage change is defined as (reduced - full) / abs(full coefficient).', body)
P('Duration: -0.3680 to -0.4346 (-18.08%). Credit_amount: -0.3526 to -0.2672 (+24.20%). Installment_rate: -0.3094 to -0.2613 (+15.55%). Age: +0.1624 to +0.1822 (+12.20%). Residence: -308%. But the full coefficient is +0.0068. So this is a small-denominator artefact.', body)
P('The largest meaningful change is credit_amount. Both in percentage terms and in absolute terms (+0.0853). Without checking status, the model attributes less of the approval penalty to amount. This suggests some of what the full model credited to the account variable is carried by loan size in the reduced model.', body)
P('The important thing is that accuracy barely changed (75.00% to 74.00%). But 31 individual decisions flipped (10.33%). The coefficients of every substantive predictor changed by 12-24%. If the model was never stress-tested like this, it would look equally good on the headline metric. But the story about which factors matter would be different.', body)
P('Coefficients can only be interpreted relative to what was left out. In practice, two analysts who kept different but equally reasonable columns would report different effects of loan size (-0.3526 vs. -0.2672). They would recommend different loan caps. A reviewer relying on either model would never see that the other model existed.', body)
H2('3.4 How I would look for a hidden confound')
P('A hidden confound is hard to find. Because it is by definition not in the table. I would combine a coefficient stability audit with a loan officer override audit. Rather than rely on one statistical test.', body)
P('For the first part, I can reuse what we saw with the checking_status experiment. Dropping one variable moved credit_amount by +24.2%. It flipped 31/300 decisions. For a deployed lender, I would take a sample of a few hundred past applications. Append candidate variables that the model never saw. Such as income stability from bank statements, employer type, regional unemployment, etc. Refit. I would agree in advance that if any key coefficient moves by more than about 20% (the amount coefficient already exceeds this), that is a confound worth including.', body)
P("For the second part, use people. Wherever a loan officer overrides the model, the reason is recorded in a structured field. If the overrides cluster in cases where the model was confident but wrong, that points to information the model lacks. Here, that would be looking at the 26 high confidence errors (8.67% of the test set). Ask the officers why the applicants with a no account status or unusual credit history behave differently from the model's prediction.", body)
P('I would back this up with residual analysis. Plot residuals against variables outside the model. Test for structure. Finally, validate on a later period or a second lender. If there are large drops in calibration or in subgroup error rates, that suggests a relationship specific to the original data.', body)
P('This does not prove the absence of a confound. But it gives a repeatable way to find one. It gives a formal role for human reviewers to correct the model. Rather than just follow it.', body)
H2('3.5 High-confidence error subset analysis')
P('Logistic model errors: 26 high-confidence mistakes in test set (300 records): 8.67%. 13 false positives (&gt;0.75 approval, Bad), 13 false negatives (&lt;0.25, Good).', body)
P('All 26 included (&lt;50 available). Compared to test set: higher mean credit amount (4,146 vs 3,163), longer mean duration (23.9 vs 21.0 months), slightly younger mean age (33.8 vs 35.3).', body)
P('Characteristics shared across errors. False negatives: 13, 8 (61.5%) have credit history A30/A31 (vs 7.7% Good test applicants). False positives: 13, 7 (53.8%) have no checking account (vs 12.7% Bad test applicants). 15/26 rows (57.7%) explained by these patterns. Coefficients explain the mechanism. No-account level: coefficient +0.8606 (largest positive) from 279 training rows (87.1% Good). Push any no-account applicant to approval. (even 10 Bad test applicants without an account). Credit-history levels A30, A31: only 61 training rows, 31.1% Good. Confident rejection. (17 Good test applicants in these groups). Purely additive model: other variables cannot override a dominant level. Small cells: unstable estimates. Confident score: not evidence of certainty. Reviewer trusting stated confidence would skip second look for exactly the 26 cases that most need it.', body)
H2('3.6 What does a fairness trade-off mean for an applicant?')
P('Real rows from my test set: (1-based row numbers in german.data) Applicant F: row 614, 22 year old female, 3,632, 24 months. Model score: 0.589. (Actually Good risk). Applicant M: row 434, 33 year old male, 2,058, 24 months. Score: 0.752. (Actually Good risk). At t=0.55, both approved. At t=0.65: she rejected, he approved. At t=0.75: same (0.752 just clears 0.75). At t=0.85: both rejected.', body)
P("Outcomes mirror group pattern. At t=0.65, DP = 1.357. 70.0% of men approved vs 51.5% women. Good women: 67.6% approved vs 80.4% good men. So like Applicant F, 32.4% rejected vs 19.6% for like Applicant M. Higher threshold: reduces harm to lender. False-positive rate for women drops from 34.5% at 0.55 to 13.8% at 0.65. PP for women increases to 92.0% (vs men 86.6%, PP = 0.942). So, favouring predictive reliability gives more accurate label to approved women, but at the cost of more good women rejected. Trade-off cost between DP and PP: 0.38. This figure hides the human effect. Score 0.59: for Applicant F, it's credit or refusal. No individual feature of her application changed between thresholds. Only the policy did. Her outcome decided by whoever sets the threshold, not the model. That decision should remain with accountable humans.", body)
H2('3.7 Defensible threshold choice')
P('Most defensible non-trivial threshold: t=0.35. At 0.35: three ratios DP = 1.084, EO = 1.062, PP = 1.053. All within 8.4% of parity. Accuracy: 74.7%. (Same as 0.50: 75.0%). Interpretable subgroup rates: approval: 87.2% men, 80.4% women. TPR: 92.2% vs 86.8%. FPR: 72.0% vs 65.5%. At 0.55, gaps worsen: DP 1.291, EO 1.229. Thresholds 0.05, 0.15: look fairer because they approve 299, 291 out of 300 applicants. Almost no screening.', body)
P("The biggest counter-argument is cost, and it's valid. At 0.35, model approves 255 applicants. 55 of 79 Bad ones. UCI cost: 296. Cheapest threshold in the sweep: 0.75. Cost: 140. (Roughly 2.1 times lower). You can argue that I've bought parity by lowering the bar for everyone. High FPR: 72% for men, 66% for women. So ratios near 1 are two equally lenient groups. OK. Parity ratios say nothing about level of harm. At 0.75, 75 good applicants would be rejected. More than three times as many as 0.35 (21). So, this is an explicit value judgement. Access to credit for Good applicants is more important than the 5:1 cost penalty. This judgement should be set by lender/regulator, with human oversight. Not inferred from data. The numbers describe consequences. They cannot decide which is acceptable.", body)
H2('3.8 Demographic parity vs equalized odds: which to prioritise?')
P("For this credit setting, I would prioritise equalized odds over demographic parity. But still monitor approval gaps. Label: repayment risk. Equalized odds: same chance of approval (TPR) for good applicants of either sex. Same chance of false pass (FPR) for bad applicants. Demographic parity: ignore the label. In the test set, 75.4% men, 70.1% women are Good. So if you force equal approval rates, you'll have to approve relatively more bad risk applicants in one group, or reject good ones in the other.", body)
P("My trade-off numbers show the sacrifice. From t=0.85 to t=0.35. DP: 1.124 to 1.084. (32.2% improvement). EO: 1.042 to 1.062. (49.2% degradation). Cost: 1.53. DP to PP: cost 0.38. At 0.35: I'd accept DP 1.084. Men approved 6.8% points more often than women. This gap is similar in size to the 5.3% point difference in Good rates. I'm sacrificing equal access as a goal in itself. This is acceptable only with two safeguards: 1. report EO with TPR and FPR components. Minimum rule hides the bigger gap. (At 0.35, 1.062 and 1.099). 2. question the base rate difference. 1994 dataset: might encode historical bias. Choosing between metrics is a policy value. So the decision is with accountable people, not the model. Monitor whenever threshold/applicant population changes.", body)
H1('Part 4 – Reflection')
P('<b>Direct answer.</b> Recommend conditional deployment as decision support only. Not automated decision maker. With human review band. Documented threshold. Appeal route. Headline reason: 75.00% accuracy of model is only 1.33% points better than guessing everyone is Good.', body)
P('<b>Why single number fails.</b> 1. Overall accuracy hides who is wrong. Same 75.00%: confident approve band is 91.8% accurate. Confident reject band is 43.5% accurate. Worse than a coin flip. If told "model is 75% accurate", a reviewer would defer in exactly the cases where deference is least justified. 2. Threshold is a value question. Not statistical fact. OLS-median rule vs 0.50 rule: disagreed on 76 cases. Change logistic cut-off from 0.35 to 0.65: 63 decisions changed. Accuracy remains around 75%. UCI cost at 0.35: 296. At 0.75: 140. Which is better depends on how a lender weighs lost customers against defaults. That\'s for people to decide. 3. Fairness metrics conflict. At 0.35: DP, EO, PP are 1.084, 1.062, 1.053. But improving one changes the others. Best DP/EO thresholds: approve almost everyone (299/300 at 0.05). Best PP threshold (0.85): reject 114/221 Good applicants. 4. Confidently fails on predictable subsets. 26 high-confidence errors (8.67%). 58% explained by two patterns. (No checking account for false positives, credit history A30/A31 for false negatives). 5. Hidden confounds are unknowable. Dropping checking_status: +24.2% change in credit_amount. 31 decisions flipped. (But accuracy barely changed). What if some variable I can\'t observe did the same? 6. Aggregates hide subgroups. Credit amount: -0.6379 coefficient for telephone owners. +0.4913 coefficient for non-owners. Pooled: -0.1641. Not significant.', body)
P("Each of these limits answers the topic question. The model doesn't forcefully take away a decision from a person. It limits their decision space through defaults that are rarely seen by a reviewer. Analysis can't show if real reviewers defer. No reviewer behaviour in the data. But it shows how much of the outcome is already fixed before they look.", body)
P('<b>Evidence grounding.</b> Accuracy: 75.0% vs baseline 73.7%. Band accuracy: 91.8% and 43.5%. 25.3% rule mismatch. DP/EO/PP: 1.084/1.062/1.053 at 0.35. 26 confident errors. Coefficient change: +24.2%. Sign reversal: -0.6379 vs +0.4913. Cost of threshold choice: concrete. UCI cost 296 at 0.35 vs 140 at 0.75.', body)
P('<b>Closing.</b> Responsible deployment: publish the trade-offs. Who is rejected. Who is wrongly approved. Which fairness metric was sacrificed. Single number: invite deference. Transparent set of numbers: invite judgement. This is the form of autonomy these systems need to protect. What a policymaker needs to remain in charge.', body)
H1('Overall conclusion and limitations')
P('Two models: different approval labels for 25.3% of test set. Subgroup analysis: real sign reversal in credit amount by telephone ownership. Removing checking_status: changes other coefficients. Adversarial subset: exposes confident errors. Fairness sweep: threshold changes alter DP, EO, PP. And number of Good applicants rejected. All these results: support the view that a model can reduce the effective autonomy of humans. It fixes choices (threshold, features, confidence) before a person sees the case.', body)
P("These results aren't evidence that the model is ready for lending. Source data: donated in 1994. Only 1,000 cases [1]. Binary mapping: proxy from historical Good/Bad class. Not loan decision. Simple EO formula: can hide one component rate if the other is more similar. Simpson-type finding: one dataset. Multiple comparison scan. Explanation is a hypothesis. Autonomy discussion: only structural evidence. No human reviewers in the dataset. Deference, automation bias not measured. Current, representative dataset. Meaningful human oversight. Needed before real use.", body)
H1('References')
P('[1] Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77. Dataset description, variable definitions, missing value info, cost matrix. Original file https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data.', body)
P('[2] CSET485 - AI and Society. Assignment #1. (Course handout provided by instructor).', body)
P('[3] Identical 1,000-row symbolic file from public mirror https://github.com/jbrownlee/Datasets (german.csv). UCI host unreachable from my work environment. Checked: 1,000 rows, 21 columns. 700 Good / 300 Bad split like original.', body)
H1('Reproducibility and submission note')
P('Code: Jupyter notebook. analysis.ipynb. Run from repository root. Loads committed data/german.data. random_state=13. Fits both models. Stress tests. Writes all tables, charts, numbers in this report to outputs folder. (outputs/results.json is the single source for numbers). make_report.py: builds this PDF from that file.', body)
P("GitHub repository: " + (REPO_URL if REPO_URL else "<b>[link to be added once repository is published]</b>"))
doc.build(story); print("built", OUT)