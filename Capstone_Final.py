#!/usr/bin/env python
# coding: utf-8

# # <center> IDS Capstone Project

# In[1]:


# Team member: Adarsh Tiwari, N-number: N17883578
# Seed used: 17883578


import numpy as np
import random

SEED = 17883578

np.random.seed(SEED)
random.seed(SEED)

# Some libraries (if used) also have their own RNG seeds
try:
    import torch
    torch.manual_seed(SEED)
except:
    pass

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except:
    pass


# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE


# #### Reading the files and assigning column names

# In[3]:


rmp_num = pd.read_csv("rmpCapstoneNum.csv", header=None)
rmp_num.columns = [
    "avg_rating",
    "avg_difficulty",
    "num_ratings",
    "pepper",
    "take_again_pct",
    "num_online_ratings",
    "male",
    "female"
]


# In[4]:


rmp_num.head()


# In[5]:


rmp_qual = pd.read_csv("rmpCapstoneQual.csv", header=None)
rmp_qual.columns = ["major_field", "university", "state"]


# In[6]:


rmp_qual.head()


# In[7]:


rmp_tags = pd.read_csv("rmpCapstoneTags.csv", header=None)
rmp_tags.columns = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation_matters",
    "dont_skip_class",
    "lots_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_by_few",
    "amazing_lectures",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy"
]


# In[8]:


rmp_tags.head()


# In[194]:


df = pd.concat([rmp_num, rmp_qual, rmp_tags], axis=1)
print(df.shape)


# ### 1. Investigating Gender Bias in Student Evaluations
# 
# Activists have asserted that there is a strong gender bias in student evaluations of professors, with male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues, skeptics have pointed out that this research is of technically poor quality, either due to a low sample size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015).
# 
# **Task:** Determine whether this dataset provides evidence of a pro-male gender bias in student evaluations.  
# 
# **Hint:** A formal significance test will likely be required.

# In[10]:


df.isnull().sum()


# In[11]:


df['graded_by_few'].value_counts()


# In[12]:


# Fill categorical NaNs
df["major_field"] = df["major_field"].fillna("Unknown")
df["university"] = df["university"].fillna("Unknown")
df["state"] = df["state"].fillna("Unknown")


numeric_cols = ["avg_rating", "avg_difficulty", "num_ratings", "num_online_ratings", "male", "female"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


# In[14]:


# Drop rows with NaN in critical numeric columns
df = df.dropna(subset=numeric_cols)


# In[15]:


df.shape


# In[16]:


# minimum rating threshold
MIN_RATINGS = 10
df = df[df["num_ratings"] >= MIN_RATINGS]


# #### Since keeping the min rating count higher resulted in loss of lots of data, trying to take threshold as 10.

# In[42]:


df.shape


# In[18]:


df_gender = df[(df["male"] == 1) | (df["female"] == 1)]


# In[20]:


male_ratings = df_gender[df_gender["male"] == 1]["avg_rating"]
female_ratings = df_gender[df_gender["female"] == 1]["avg_rating"]


# In[21]:


# 5. Perform Welch's t-test

tstat, pval = ttest_ind(male_ratings, female_ratings, equal_var=False)

print("Male mean rating:", male_ratings.mean())
print("Female mean rating:", female_ratings.mean())
print("t-statistic:", tstat)
print("p-value:", pval)


# ### Analysis of T-test Results
# 
# **Problems with the t-test:**
# 
# - I have ignored confounders such as class difficulty, number of ratings, online ratings or subject/department.
# - Unequal sample sizes and variance between male/female professors may bias the result.
# - Small or noisy ratings can exaggerate differences.
# 
# **Conclusion:**  
# The t-test shows a statistically significant difference between male and female professors' ratings, but it **does not prove a causal gender effect**. The effect is likely because of the above mentioned issues

# In[23]:


df2 = df[(df["num_ratings"] >= 10) & ((df.male == 1) | (df.female == 1))]


# In[24]:


df2["female"] = df2["female"].astype(int)
df2["male"] = df2["male"].astype(int)

X = df2[["male"]]  # main predictor
X = sm.add_constant(X)
y = df2["avg_rating"]

model = sm.OLS(y, X).fit()
print(model.summary())


# 
# **Key observations:**
# 
# This simple OLS regression examines whether a professor being male is associated with differences in **average rating**.
# 
# #### Model Fit
# - **R² = 0.002**: The model explains only **0.2%** of the variation in average ratings.
# - Despite statistical significance, the explanatory power is **extremely small**.
# - Large sample size (**N = 7,406**) makes even very small effects statistically detectable.
# 
# #### Coefficient Interpretation
# - **Intercept (3.889)**: Expected average rating for the baseline group (non-male).
# - **Male coefficient (0.075)**:
#   - Male professors receive, on average, **0.075 higher rating points** than non-male professors.
#   - The effect is **statistically significant** (p < 0.001).
#   - However, the magnitude is **practically negligible** on a 1–5 rating scale.
# 
# #### Conclusion
# While gender (male) shows a statistically significant association with average rating, it has **no meaningful predictive value** on its own. Substantive differences in ratings are driven by other factors (e.g., difficulty, teaching quality, and tags), not gender alone.
# 

# #### Taking confounders under consideration

# In[29]:


df.shape


# In[30]:


df_reg = df[(df["male"] == 1) | (df["female"] == 1)].copy()
df_reg["male_flag"] = (df_reg["male"] == 1).astype(int)
df_reg["online_frac"] = pd.to_numeric(df_reg["num_online_ratings"], errors='coerce') / pd.to_numeric(df_reg["num_ratings"], errors='coerce')


# Numeric confounders
top_majors = df_reg['major_field'].value_counts().nlargest(10).index
df_reg['major_reduced'] = df_reg['major_field'].where(df_reg['major_field'].isin(top_majors), 'Other')


df_dummies = pd.get_dummies(df_reg, columns=['major_reduced', 'state', 'university'], drop_first=True)


# In[31]:


#preparing model for regression

y = pd.to_numeric(df_dummies["avg_rating"], errors='coerce')

predictors = ["male_flag", "avg_difficulty", "take_again_pct", "online_frac"] + \
             [col for col in df_dummies.columns if col.startswith(("major_reduced_", "university_reduced_", "state_reduced_"))]

X = df_dummies[predictors].astype(float)
y = y.astype(float)

# Drop any rows with NaNs
valid_idx = y.notna() & X.notna().all(axis=1)
y_clean = y[valid_idx]
X_clean = sm.add_constant(X.loc[valid_idx])


# In[32]:


model = sm.OLS(y_clean, X_clean).fit(cov_type='HC3')
print(model.summary())


# In[34]:


sns.boxplot(x='male_flag', y='avg_rating', data=df_reg)
plt.xticks([0,1], ['Female', 'Male'])
plt.ylabel('Average Rating')
plt.title('Average Rating by Gender')
plt.show()


# We tested whether male professors receive higher ratings than female professors after controlling for important confounders.
# 
# **Predictors included:**
# - Male gender (`male_flag`)
# - Average difficulty (`avg_difficulty`)
# - Proportion of students who would take the class again (`take_again_pct`)
# - Fraction of online ratings (`online_frac`)
# - Major (top 10 majors + "Other")
# 
# **Key Results:**
# 
# | Predictor        | Coefficient | P-value | Interpretation |
# |-----------------|------------|---------|----------------|
# | male_flag        | 0.0084     | 0.384   | Not significant; no evidence of pro-male bias |
# | avg_difficulty   | -0.2308    | <0.001  | Harder courses are rated lower |
# | take_again_pct   | 0.0258     | <0.001  | Courses students would take again are rated higher |
# | online_frac      | 0.0164     | 0.638   | Slightly higher ratings for online courses, not significant |
# 
# - R² = 0.798 → model explains most of the variance in ratings.  
# - Condition number ≈ 1200 → mild multicollinearity among major dummies.  
# 
# **Conclusion:**  
# After accounting for confounders (course difficulty, take-again %, online fraction, and major) the gender effect on average ratings is **negligible and not statistically significant**. Observed differences are largely explained by these confounders rather than gender itself.

# ### 2. Is there a gender difference in the spread (variance/dispersion) of the ratings distribution? Again, it is advisable to consider the statistical significance of any observed gender differences in this spread.

# In[51]:


male_ratings = df.loc[df['male'] == 1, 'avg_rating']
female_ratings = df.loc[df['female'] == 1, 'avg_rating']


# In[52]:


male_var = male_ratings.var()
female_var = female_ratings.var()

print(f"Male variance: {male_var:.4f}")
print(f"Female variance: {female_var:.4f}")


# In[53]:


if male_var >= female_var:
    F = male_var / female_var
    df1 = len(male_ratings) - 1
    df2 = len(female_ratings) - 1
else:
    F = female_var / male_var
    df1 = len(female_ratings) - 1
    df2 = len(male_ratings) - 1

print(f"F-statistic: {F:.3f}")


# In[54]:


pval = 2 * min(f.cdf(F, df1, df2), 1 - f.cdf(F, df1, df2))
print(f"P-value: {pval:}")


# In[55]:


alpha = 0.005
if pval < alpha:
    print("Statistically significant difference in variance between male and female ratings.")
else:
    print("No statistically significant difference in variance between male and female ratings.")


# ### Gender Difference in Ratings Spread (F-test)
# 
# - **Male variance:** 0.735 
# - **Female variance:** 0.801 
# - **F-statistic:** 1.09  
# - **P-value:**  0.007638081204012437 
# 
# **Conclusion:**  
# 
# At
# α=0.005, the p-value 0.00763 > 0.005, so we fail to reject the null hypothesis. There is no statistically significant difference in rating dispersion between male and female professors at this stricter significance level..

# ### Question 3: Effect Size of Gender Bias
# 
# We aim to estimate the **likely size of gender effects** in this dataset:
# 
# 1. **Gender bias in average rating:**  
#    Estimate the difference in mean ratings between male and female professors, with **95% confidence interval**.
# 
# 2. **Gender bias in spread of ratings:**  
#    Estimate the difference in **variances or standard deviations** between male and female professors, with **95% confidence interval**.
# 

# In[59]:


# male and female ratings
male_ratings = df[df['male'] == 1]['avg_rating']
female_ratings = df[df['female'] == 1]['avg_rating']

cm = sms.CompareMeans(sms.DescrStatsW(male_ratings), sms.DescrStatsW(female_ratings))
mean_diff = male_ratings.mean() - female_ratings.mean()
ci_mean_diff = cm.tconfint_diff(usevar='unequal')

print("Mean difference (male - female):", mean_diff)
print("95% CI for mean difference:", ci_mean_diff)


# In[60]:


means = [male_ratings.mean(), female_ratings.mean()]
stds = [male_ratings.std(), female_ratings.std()]
labels = ['Male', 'Female']

# Plot
plt.figure(figsize=(6,5))
plt.bar(labels, means, yerr=stds, capsize=8, color=['skyblue','lightcoral'])
plt.ylabel('Average Rating')
plt.title('Average Rating ± SD by Gender')
plt.show()


# In[61]:


male_var = male_ratings.var()
female_var = female_ratings.var()
std_diff = male_ratings.std() - female_ratings.std()

# Approximate 95% CI for variance ratio using F-distribution
alpha = 0.05
f_lower = stats.f.ppf(alpha/2, male_ratings.size-1, female_ratings.size-1)
f_upper = stats.f.ppf(1-alpha/2, male_ratings.size-1, female_ratings.size-1)
var_ratio = male_var / female_var
ci_lower = var_ratio / f_upper
ci_upper = var_ratio / f_lower

print("Variance ratio (male/female):", var_ratio)
print("Approx. 95% CI for variance ratio:", (ci_lower, ci_upper))
print("Difference in standard deviation (male - female):", std_diff)


# In[62]:


sns.boxplot(x=df['male'].replace({1:'Male', 0:'Female'}),
            y=df['avg_rating'],
            palette=['skyblue','lightcoral'])
plt.ylabel('Average Rating')
plt.xlabel('Gender')
plt.title('Spread of Ratings by Gender')
plt.show()


# ### Gender Differences in Ratings
# 
# | Metric | Value | 95% Confidence Interval | Interpretation |
# |--------|-------|------------------------|----------------|
# | Mean difference (male - female) | 0.0689 | (0.0294, 0.1084) | Male professors have slightly higher average ratings |
# | Variance ratio (male/female) | 0.9173 | (0.8607, 0.9774) | Variance of male ratings slightly lower than female ratings |
# | Difference in standard deviation (male - female) | -0.0378 | — | Spread of ratings is very similar |
# 
# **Key Insight:**  
# While male professors appear to have slightly higher ratings, the difference is small. The spread of ratings is nearly equal between genders, indicating no meaningful difference in dispersion.

# ### Question 4: Gender Difference in Tags
# 
# Is there a gender difference in the tags awarded by students? Analyze all 20 tags for potential gender differences and report which tags exhibit statistically significant differences. Comment on the three most gendered (lowest p-value) and three least gendered (highest p-value) tags.

# In[67]:


rmp_tags_numeric = df.select_dtypes(include='number')

num_ratings_safe = rmp_tags_numeric['num_ratings'].replace(0, 1)

rmp_tags_norm = rmp_tags_numeric.div(num_ratings_safe, axis=0)

results = []

# male vs female for each tag
for tag in rmp_tags.columns:
    male_values = rmp_tags_norm[tag][rmp_num['male']==1]
    female_values = rmp_tags_norm[tag][rmp_num['female']==1]
    
    stat, pval = ttest_ind(male_values, female_values, equal_var=False)  # Welch's t-test
    results.append((tag, stat, pval))


# In[68]:


tag_results = pd.DataFrame(results, columns=['Tag','t_stat','p_value'])
tag_results = tag_results.sort_values('p_value')

significant_tags = tag_results[tag_results['p_value'] < 0.05]

# Top 3 most gendered and top 3 least gendered
most_gendered = tag_results.head(3)
least_gendered = tag_results.tail(3)

print("Significant tags:\n", significant_tags)
print("\nThree most gendered tags:\n", most_gendered)
print("\nThree least gendered tags:\n", least_gendered)


# **Approach:**  
# - Normalized tag counts by each professor’s number of ratings to adjust for exposure.  
# - Grouped professors by gender (`male==1` vs `female==1`).  
# - Applied Welch’s t-test for each tag to compare male vs female professors, accounting for unequal variances.  
# 
# **Findings:**  
# - **Significant tags (p < 0.05):** Multiple tags show gender differences in frequency of being awarded.  
# 
# - **Three most gendered tags:**  
# 
# | Tag               | t_stat     | p_value       | Interpretation                                 |
# |------------------|------------|---------------|-----------------------------------------------|
# | hilarious         | 17.744203  | 4.48e-69      | Strongly awarded more to male professors     |
# | caring            | -9.154416  | 7.05e-20      | Strongly awarded more to female professors   |
# | amazing_lectures  | 8.576727   | 1.17e-17      | Awarded more to male professors              |
# 
# - **Three least gendered tags:**  
# 
# | Tag           | t_stat    | p_value   | Interpretation                             |
# |---------------|-----------|-----------|-------------------------------------------|
# | accessible    | 2.879001  | 0.004001  | Slight male bias                            |
# | tough_grader  | -2.506731 | 0.012207  | Slight female bias                          |
# | pop_quizzes   | -0.802636 | 0.422211  | No meaningful gender difference            |
# 
# **Key Insight:**  
# Although a few tags show strong gendered patterns, most tags exhibit minor or no bias. Students selectively apply gendered labels, suggesting specific stereotypes rather than uniform bias across all tags.

# ### Question 5: Gender Difference in Average Difficulty
# 
# Is there a gender difference in terms of average difficulty ratings of professors? Perform a statistical significance test to determine whether male and female professors are rated differently in terms of difficulty.

# In[69]:


male_diff = df.loc[df['male'] == 1, 'avg_difficulty']
female_diff = df.loc[df['female'] == 1, 'avg_difficulty']


# In[72]:


male_var = male_diff.var()
female_var = female_diff.var()
print(f"Male variance: {male_var:.4f}")
print(f"Female variance: {female_var:.4f}")


# In[73]:


t_stat, p_value = ttest_ind(male_diff, female_diff, equal_var=False)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")


# In[74]:


diff_mean = male_diff.mean() - female_diff.mean()
se_diff = np.sqrt(male_diff.var()/len(male_diff) + female_diff.var()/len(female_diff))
ci_lower = diff_mean - 1.96 * se_diff
ci_upper = diff_mean + 1.96 * se_diff
print(f"Mean difference (male - female): {diff_mean:.4f}")
print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")


# In[75]:


sns.boxplot(x='male', y='avg_difficulty', data=df.replace({'male': {0:'Female',1:'Male'}}))
plt.title("Average Difficulty by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Difficulty")
plt.show()


# ### Question 6: Quantifying the Gender Difference in Average Difficulty
# 
# 
# 
# 
# 
# 
# 
# 

# In[76]:


# 95% CI for mean difference
n1 = len(male_diff)
n2 = len(female_diff)
s1 = male_diff.var(ddof=1)
s2 = female_diff.var(ddof=1)

# Standard error
se = np.sqrt(s1/n1 + s2/n2)

# 95% CI (Welch)
ci_low = diff_mean - 1.96 * se
ci_high = diff_mean + 1.96 * se

print("Mean difference (male - female):", diff_mean)
print("95% CI:", (ci_low, ci_high))


# ### Question 6: Likely Size of Gender Effect on Average Difficulty
# 
# | Metric | Value | 95% Confidence Interval | Interpretation |
# |--------|-------|-----------------------|----------------|
# | Mean difference (male - female) | -0.0048 | (-0.0393, 0.0296) | Effect is negligible |
# 
# **Key Insight:**  
# The likely size of the gender effect on average difficulty is **extremely small** with the confidence interval including zero. There is **no meaningful difference** between male and female professors in perceived difficulty.

# ### Question 7: Build a regression model predicting average rating from all numerical predictors (the ones in the rmpCapstoneNum.csv file).  
# Include the R² and RMSE of this model.  
# Which of these factors is most strongly predictive of average rating?  
# (Hint: Make sure to address collinearity concerns.)

# In[77]:


num_cols = ["avg_difficulty", "num_ratings", "pepper", "take_again_pct", "num_online_ratings", "male", "female"]

X = df[num_cols]
y = df["avg_rating"]
# Constant for intercept
X = sm.add_constant(X)


# In[78]:


print(X.isna().sum())

# Check for infinite values
print(np.isinf(X).sum())


# In[79]:


print(y.isna().sum())

# Check for infinite values
print(np.isinf(y).sum())


# In[92]:


# Original y without NaNs
y_original = y.dropna()


# In[93]:


print("Original y stats:\n", y_original.describe())


# In[94]:


X_work = X.copy()


# In[95]:


# Skewed / percentage column
X_work["take_again_pct"] = X_work["take_again_pct"].fillna(X_work["take_again_pct"].median())

# Other numeric columns
for col in ["avg_difficulty", "num_ratings", "num_online_ratings"]:
    X_work[col] = X_work[col].fillna(X_work[col].mean())

# pepper is boolean, fill with 0 (not peppered)
X_work["pepper"] = X_work["pepper"].fillna(0)
X_work_const = sm.add_constant(X_work)


# In[96]:


plt.figure(figsize=(20, 12))

for i, col in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)  # Adjust rows/columns as needed
    sns.kdeplot(X[col].dropna(), label='Original', color='blue')
    sns.kdeplot(X_work_const[col], label='Processed', color='orange')
    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.show()


# In[97]:


print(X_work_const.isna().sum())

# Check for infinite values
print(np.isinf(X_work_const).sum())


# In[98]:


vif_data = pd.DataFrame()
vif_data["feature"] = X_work_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_work_const.values, i) for i in range(X_work_const.shape[1])]
print(vif_data)


# In[99]:


X_work_const.head()


# In[100]:


X_work_const.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[90]:


X_work_clean = X_work_const.drop(columns=['female'])


# In[101]:


X_reg = X_work_clean.copy() 
y_reg = y_original  

#  OLS model
model2 = sm.OLS(y_reg, X_reg).fit()
print(model2.summary())


# In[89]:


coefs = model2.params.drop('const')
plt.figure(figsize=(12,6))
sns.barplot(x=coefs.index, y=coefs.values, palette='viridis')
plt.xticks(rotation=45)
plt.ylabel("Standardized Coefficient")
plt.title("Predictive Strength of Numerical Features on Average Rating")
plt.show()


# ### Question 8: 
# 
# Build a regression model predicting **average rating** from all **tags** (from `rmpCapstoneTags.csv`). Include **R²** and **RMSE**. Identify the tag most strongly predictive of average rating, account for **collinearity**, and compare this model to the previous regression with numerical predictors.

# In[134]:


rmp_tags.columns


# In[135]:


num_cols = ['tough_grader', 'good_feedback', 'respected', 'lots_to_read',
       'participation_matters', 'dont_skip_class', 'lots_homework',
       'inspirational', 'pop_quizzes', 'accessible', 'many_papers',
       'clear_grading', 'hilarious', 'test_heavy', 'graded_by_few',
       'amazing_lectures', 'caring', 'extra_credit', 'group_projects',
       'lecture_heavy']

df_tags = df[num_cols]
y_tags = df['avg_rating'].copy()


# In[137]:


print(df_tags.isna().sum())
print(y_tags.isna().sum())


# In[139]:


X = pd.get_dummies(df_tags, drop_first=True)


# In[140]:


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

mask = y_tags.notna()
X_A = X[mask]
y_A = y_tags[mask]


# In[141]:


scaler_A = StandardScaler()
X_A_scaled = scaler_A.fit_transform(X_A)


# In[142]:


model_A = LinearRegression()
model_A.fit(X_A_scaled, y_A)

# Predict
y_pred_A = model_A.predict(X_A_scaled)


# In[143]:


rmse_A = mean_squared_error(y_A, y_pred_A) ** 0.5
r2_A = r2_score(y_A, y_pred_A)
print("R²:", r2_A)
print("RMSE:", rmse_A)


# In[144]:


# Coefficients with names
coef_A = pd.DataFrame({
    "feature": X_A.columns,
    "coef": model_A.coef_
})

# Sort by absolute impact
coef_A_sorted = coef_A.reindex(coef_A.coef.abs().sort_values(ascending=False).index)
print("\nTop predictive tags:")
print(coef_A_sorted.head(10))


# In[146]:


top_tags = coef_A_sorted.reindex(coef_A_sorted['coef'].abs().sort_values(ascending=False).index).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x='coef', y='feature', data=top_tags, palette='coolwarm')
plt.title("Top 10 Predictive Tags for Average Rating")
plt.xlabel("Coefficient")
plt.ylabel("Tag")
plt.axvline(0, color='black', linewidth=0.8)
plt.show()


# # Tags Predicting Average Rating (dropping null y's)
# 
# **Top Predictive Tags:**  
# 1. `tough_grader` (negative, -0.348) → courses tagged as tough grader tend to have lower ratings  
# 2. `amazing_lectures` (positive, 0.144) → higher ratings  
# 3. `lecture_heavy` (negative, -0.135) → lower ratings  
# 4. `good_feedback` (positive, 0.133)  
# 5. `caring` (positive, 0.114)  
# 
# **Comparison to Previous Model (Numerical Predictors):**  
# - Tags-only model explains less variance (R² ~0.18) than the model with numerical predictors (R² ~0.37).  
# - Certain tags capture nuanced effects, but course metrics remain stronger predictors of average rating.

# **Comparison with Question 7 (Numerical Predictors) vs Question 8 (Tags)**
# 
# - Model using numerical predictors (Question 7) achieved much higher R² (~0.37) and lower RMSE (~0.99) compared to the tags-only model (Question 8) with R² ~0.18 and RMSE ~0.90.  
# - This indicates that numerical features (difficulty, ratings count, pepper, take-again %, etc.) are more strongly predictive of average rating than the 20 tags.  
# - Top predictive tags (e.g., tough_grader, good_feedback, caring) show meaningful but smaller effect sizes compared to numerical predictors.  
# - Tags provide interpretability and qualitative insight but add less predictive power than quantitative features.

# ### **Question 9**
# 
# Build a regression model predicting **average difficulty** from all tags (the ones in `rmpCapstoneTags.csv`). Include **R²** and **RMSE** of this model. Identify which tags are most strongly predictive of average difficulty. Make sure to check and address **collinearity concerns**.

# In[152]:


num_cols = ['tough_grader', 'good_feedback', 'respected', 'lots_to_read',
       'participation_matters', 'dont_skip_class', 'lots_homework',
       'inspirational', 'pop_quizzes', 'accessible', 'many_papers',
       'clear_grading', 'hilarious', 'test_heavy', 'graded_by_few',
       'amazing_lectures', 'caring', 'extra_credit', 'group_projects',
       'lecture_heavy']

X_tags = df[num_cols]
y_diff = df['avg_difficulty']


# In[153]:


print(X_tags.isna().sum())
print(np.isinf(X_tags).sum())


# In[154]:


mask_diff = y_diff.notna()
X_A_diff = X_tags[mask_diff]
y_A_diff = y_diff[mask_diff]


# In[155]:


scaler_A_diff = StandardScaler()
X_A_scaled_diff = scaler_A_diff.fit_transform(X_A_diff)


# In[156]:


model_A_diff = LinearRegression()
model_A_diff.fit(X_A_scaled_diff, y_A_diff)

# Predictions
y_pred_A_diff = model_A_diff.predict(X_A_scaled_diff)

# Metrics
r2_A_diff = r2_score(y_A_diff, y_pred_A_diff)
rmse_A_diff = np.sqrt(mean_squared_error(y_A_diff, y_pred_A_diff))


# In[157]:


coef_A_diff = pd.DataFrame({
    'feature': X_A_diff.columns,
    'coef': model_A_diff.coef_
}).sort_values(by='coef', key=abs, ascending=False)

print(f"R²: {r2_A_diff}")
print(f"RMSE: {rmse_A_diff}")
print(coef_A_diff.head(10))


# In[158]:


top_tags_diff = coef_A_diff.copy()
top_tags_diff['abs_coef'] = top_tags_diff['coef'].abs()
top_tags_diff = top_tags_diff.sort_values('abs_coef', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x='coef', y='feature', data=top_tags_diff, palette='coolwarm')
plt.title("Top 10 Tag Coefficients Predicting Average Difficulty")
plt.xlabel("Coefficient")
plt.ylabel("Tag")
plt.show()


# ### Predicting Average Difficulty from Tags
# 
# **Key Findings:**
# - **Strongest positive predictor:** `tough_grader` (0.348) → higher difficulty.
# - **Strongest negative predictors:** `caring` (-0.101), `clear_grading` (-0.124) → lower difficulty.
# - Other tags have minor effects on difficulty.

# In[68]:


y_diff_imputed = y_diff.fillna(y_diff.median())

# Mask is now unnecessary since we imputed
X_B_scaled = scaler_A.fit_transform(X_tags)

model_B = LinearRegression()
model_B.fit(X_B_scaled, y_diff_imputed)

# Predictions
y_pred_B = model_B.predict(X_B_scaled)



# In[70]:


# Metrics
r2_B = r2_score(y_diff_imputed, y_pred_B)
rmse_B = mean_squared_error(y_diff_imputed, y_pred_B) ** 0.5


# In[71]:


# Coefficients
coef_B = pd.DataFrame({'feature': X_tags.columns, 'coef': model_B.coef_})
coef_B_sorted = coef_B.reindex(coef_B['coef'].abs().sort_values(ascending=False).index)

# Print results
print("R²:", r2_B)
print("RMSE:", rmse_B)
coef_B_sorted.head(10)


# In[72]:


plt.figure(figsize=(12,6))
sns.barplot(x='coef', y='feature', data=coef_B_sorted.head(10), palette='viridis')
plt.title("Top 10 Tag Predictors of Average Difficulty")
plt.xlabel("Coefficient")
plt.ylabel("Tag")
plt.show()


# # Question 10 - 
# Build a classification model that predicts whether a professor receives a “pepper” from all available
# factors (both tags and numerical). Make sure to include model quality metrics such as AU(RO)C and
# also address class imbalance concerns.

# In[166]:


sns.countplot(x=df['pepper'])
plt.title("Distribution of 'Pepper' (Target Variable)")
plt.show()


# In[167]:


num_cols = ['avg_rating', 'avg_difficulty', 'num_ratings', 
            'take_again_pct', 'num_online_ratings', 'male', 'female']  # adjust if needed
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['pepper'], y=rmp_num[col])
    plt.title(f"{col} vs Pepper")
    plt.show()


# In[175]:


# Identify numeric columns only
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Fill missing numerical values with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# For binary tag columns (0/1), fill missing with 0
tag_cols = rmp_tags.columns
df[tag_cols] = df[tag_cols].fillna(0)

# For categorical/string columns (like major), you can encode or drop
cat_cols = df.select_dtypes(include=['object']).columns
# Option 1: Drop for now
df2 = df.drop(columns=cat_cols)
# Option 2: Encode using one-hot if you want to keep them
# df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Check remaining NaNs
df2.isna().sum()


# In[176]:


from sklearn.decomposition import PCA

X = df.drop(columns=['pepper'])
y = df['pepper']
mask = y.notna()

X = X[mask]
y = y[mask]


# In[182]:


# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[183]:


# Step 6: PCA for dimensionality reduction (retain 95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original features: {X.shape[1]}, After PCA: {X_train_pca.shape[1]}")


# In[184]:


rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_rf_proba = rf.predict_proba(X_test)[:,1]
print("Random Forest AUROC:", roc_auc_score(y_test, y_rf_proba))
print(classification_report(y_test, rf.predict(X_test)))


# In[187]:


y_rf_pred = rf.predict(X_test)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_rf_pred)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=[0,1])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()


# In[186]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_rf_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.show()


# ### Random Forest Model: Predicting “Pepper”
# 
# ### Random Forest Classification for “Pepper” Prediction
# 
# **Approach:**  
# Random Forest classifier used to predict whether a professor receives a “pepper” based on numerical features and tags. Class imbalance handled with `class_weight='balanced'`.
# 
# **Key Metrics:**  
# - **AUROC:** 0.805 → good discrimination between classes  
# - **Accuracy:** 72%  
# - **Precision / Recall / F1-score:**  
#   - Class 0 (no pepper): Precision 0.74, Recall 0.71, F1 0.73  
#   - Class 1 (pepper): Precision 0.71, Recall 0.74, F1 0.72  
# 
# **Interpretation:**  
# The model predicts “pepper” fairly accurately. AUROC > 0.8 indicates good class separation, and performance is balanced across both classes.

# In[188]:


# 1. Initialize Logistic Regression with balanced class weights
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')


# In[189]:


logreg.fit(X_train, y_train)
y_log_proba = logreg.predict_proba(X_test)[:, 1]


y_log_pred = logreg.predict(X_test)


print("Logistic Regression AUROC:", roc_auc_score(y_test, y_log_proba))

print(classification_report(y_test, y_log_pred))


# In[190]:


cm_log = confusion_matrix(y_test, y_log_pred)
disp_log = ConfusionMatrixDisplay(cm_log, display_labels=[0, 1])
disp_log.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# In[191]:


fpr, tpr, thresholds = roc_curve(y_test, y_log_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc="lower right")
plt.show()


# ### Logistic Regression Classification for “Pepper” Prediction
# 
# **Approach:**  
# Logistic Regression used to predict whether a professor receives a “pepper” based on numerical features and tags. Class imbalance addressed.
# 
# **Key Metrics:**  
# - **AUROC:** 0.814 → good discrimination between classes  
# - **Accuracy:** 73%  
# - **Precision / Recall / F1-score:**  
#   - Class 0 (no pepper): Precision 0.76, Recall 0.70, F1 0.73  
#   - Class 1 (pepper): Precision 0.71, Recall 0.77, F1 0.74  
# 
# **Interpretation:**  
# The model shows good predictive ability, slightly better than Random Forest in AUROC and accuracy. Performance is balanced across both classes.

# ## Extra Credit
# 

# In[195]:


df = pd.concat([rmp_num, rmp_qual, rmp_tags], axis=1)
df["major_field"] = df["major_field"].fillna("Unknown")
df["university"] = df["university"].fillna("Unknown")
df["state"] = df["state"].fillna("Unknown")


numeric_cols = ["avg_rating", "avg_difficulty", "num_ratings", "num_online_ratings", "male", "female"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=numeric_cols)

MIN_RATINGS = 10
df = df[df["num_ratings"] >= MIN_RATINGS]


# In[197]:


df.columns


# In[200]:


# Does "Hotness" vary by state?
df_state = df[['state', 'pepper']].dropna()

# Compute state-wise hotness
state_hotness = (
    df_state
    .groupby('state')['pepper']
    .mean()
    .reset_index()
    .rename(columns={'pepper': 'hotness_rate'})
    .sort_values('hotness_rate', ascending=False)
)

state_hotness.head(10)


# In[202]:


top_courses = (
    df.groupby("major_field")["num_ratings"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
top_courses.plot(kind="bar")
plt.title("Top 10 Most Chosen Courses")
plt.xlabel("Course")
plt.ylabel("Total Number of Ratings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[203]:


tough_courses_diff = (
    df.groupby("major_field")["avg_difficulty"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
tough_courses_diff.plot(kind="bar")
plt.title("Top 10 Courses with Highest Average Difficulty")
plt.xlabel("Course")
plt.ylabel("Average Difficulty")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ## Clustering Professors into teaching types

# In[205]:


tag_cols = [
    'tough_grader', 'good_feedback', 'respected', 'lots_to_read',
    'participation_matters', 'dont_skip_class', 'lots_homework',
    'inspirational', 'pop_quizzes', 'accessible', 'many_papers',
    'clear_grading', 'hilarious', 'test_heavy', 'graded_by_few',
    'amazing_lectures', 'caring', 'extra_credit',
    'group_projects', 'lecture_heavy'
]

X = df[tag_cols]

X = X.fillna(0)


# In[206]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[208]:


from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    n_init=20
)

df['teaching_type'] = kmeans.fit_predict(X_scaled)


# In[209]:


cluster_profiles = (
    df.groupby('teaching_type')[tag_cols]
      .mean()
      .round(2)
)

cluster_profiles


# In[210]:


teaching_type_labels = {
    0: "Strict & Tough Graders",
    1: "Engaging & Inspirational",
    2: "Clear & Organized",
    3: "Supportive & Caring",
    4: "Workload-Heavy"
}

df['teaching_type_label'] = df['teaching_type'].map(teaching_type_labels)


# In[229]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

# Get unique labels
unique_labels = df['teaching_type_label'].unique()
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Plot each cluster with its label
for i, label in enumerate(unique_labels):
    plt.scatter(
        X_pca[df['teaching_type_label'] == label, 0],
        X_pca[df['teaching_type_label'] == label, 1],
        alpha=0.6,
        color=colors[i],
        label=label
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Professor Teaching Style Clusters (k=5)")
plt.legend(title="Teaching Types")
plt.show()


# In[223]:


# KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
df['Teaching_Type'] = kmeans.fit_predict(X_scaled)


# In[226]:


df.columns


# In[227]:


for k in range(5):
    cluster = df[df['Teaching_Type'] == k]
    
    top_tags = cluster[tag_cols].mean().nlargest(3)
    avg_rating = cluster['avg_rating'].mean()
    avg_difficulty = cluster['avg_difficulty'].mean()
    
    print(f"Teaching Type {k+1}")
    print(" Top traits:", [t.replace('_', ' ') for t in top_tags.index])
    print(f" Avg rating: {avg_rating:.2f}")
    print(f" Avg difficulty: {avg_difficulty:.2f}")
    print(f" N = {len(cluster)}\n")


# In[ ]:




