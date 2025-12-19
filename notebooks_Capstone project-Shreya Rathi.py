import pandas as pd

df_num_col = ['Avg_Quality_Rating', 'Avg_Difficulty_Rating', 'Num_Ratings', 'Has_Pepper', 'Take_Course_Again', 'Num_Online_Ratings', 'Is_Male', 'Is_Female']

df_qual_col = ['Major_Field', 'University', 'US_State']

df_tags_col = ['Tough_Grader', 'Good_Feedback', 'Respected', 'Lots_To_Read',
    'Participation_Matters', 'Dont_Skip_Class', 'Lots_Of_Homework', 'Inspirational', 'Pop_Quizzes', 'Accessible', 'So_Many_Papers', 'Clear_Grading', 'Hilarious', 'Test_Heavy', 'Graded_By_Few', 'Amazing_Lectures', 'Caring', 'Extra_Credit', 'Group_Projects',
    'Lecture_Heavy']

df_num = pd.read_csv('rmpCapstoneNum.csv', header=None, names=df_num_col)
df_qual = pd.read_csv('rmpCapstoneQual.csv', header=None, names=df_qual_col)
df_tags = pd.read_csv('rmpCapstoneTags.csv', header=None, names=df_tags_col)


df_num.head(5)

df_num.shape

df_qual.head(5)

df_qual.shape

df_tags.head(5)

df_tags.shape

combined_df = pd.concat([df_num, df_qual, df_tags], axis=1)

combined_df.head(5)

combined_df.shape

import numpy as np

N_NUMBER = 14048293
np.random.seed(N_NUMBER)
print(f"Random seed set to : {N_NUMBER}")

combined_df.isnull().sum()

# To decide threshold value
print(combined_df['Num_Ratings'].describe())

# Threshold selection based on Distribution Analysis
# Since 50% of professors have <= 3 ratings, we chose N=5
combined_df = combined_df[combined_df['Num_Ratings'] >= 3]

combined_df.shape

combined_df.isnull().sum()

# Professor with 0 or NaN total ratings
initial_rows = len(combined_df)
combined_df.dropna(subset=['Num_Ratings'], inplace=True)
combined_df = combined_df[combined_df['Num_Ratings'] > 0]
print(f"Dropped rows with 0 or NaN total ratings. Rows removed: {initial_rows - len(combined_df)}") 

combined_df.dropna(subset=['University', 'Major_Field', 'Num_Ratings'], inplace=True)

# Replace averages with Median because The median (the middle value of a sorted dataset) is preferred over the mean (average) for imputation because it is robust to outliers. If a column has a few extremely high or low values, the median remains a more representative, unbiased central tendency for the majority of the data.
cols = ['Avg_Quality_Rating', 'Avg_Difficulty_Rating', 'Take_Course_Again']
combined_df[cols] = combined_df[cols].fillna(combined_df[cols].median())

""" Replace Tags and Booleans with 0
combined_df[df_tags_col] = combined_df[df_tags_col].fillna(0)
combined_df[['Has_Pepper', 'Is_Male', 'Is_Female', 'Num_Online_Ratings']] = combined_df[['Has_Pepper', 'Is_Male', 'Is_Female', 'Num_Online_Ratings']].fillna(0)
 """

""" Consistency for Clip proportions and resolve gender conflicts
combined_df['Take_Course_Again'] = np.clip(combined_df['Take_Course_Again'], 0, 1)
combined_df.loc[(combined_df['Is_Male'] == 1) & (combined_df['Is_Female'] == 1), ['Is_Male', 'Is_Female']] = 0
"""

# For que Q1 to Q6. Only include records where gender is clearly defined as M or F
gender_df = combined_df[((combined_df['Is_Male'] == 1) | (combined_df['Is_Female'] == 1)) & ~((combined_df['Is_Male'] == 1) & (combined_df['Is_Female'] == 1))].copy()
male_ratings = gender_df[gender_df['Is_Male'] == 1]['Avg_Quality_Rating']
female_ratings = gender_df[gender_df['Is_Female'] == 1]['Avg_Quality_Rating']

combined_df.shape

combined_df.isnull().sum()

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Q1: Gender Bias in Rating
t_stat, p_val_q1 = stats.ttest_ind(male_ratings, female_ratings)

print({p_val_q1})

if p_val_q1 < 0.005:
    if male_ratings.mean() > female_ratings.mean():
        print("Yes, Student ratings are biased towards male professors")
    else:
        print("No, Student ratings are biased towards female professors")
else:
    print("No, The Student ratings are not biased towards any gender professors")

# Q2: Spread (Variance)
levene_stat, p_val_q2 = stats.levene(male_ratings, female_ratings)

if p_val_q2 < 0.005:
    if np.std(male_ratings) != np.std(female_ratings):
        print("Yes, Significant difference in the spread of ratings.")
else:
    print("No, No significant difference in the spread (variance) of ratings.")

# Q3: Bootstrapping for 95% CI of Mean and Spread Differences
def get_ci(data1, data2, iterations=1000):
    mean_diffs, std_diffs = [], []
    for _ in range(iterations):
        s1 = np.random.choice(data1, len(data1), replace=True)
        s2 = np.random.choice(data2, len(data2), replace=True)
        mean_diffs.append(np.mean(s1) - np.mean(s2))
        std_diffs.append(np.std(s1) - np.std(s2))
    return np.percentile(mean_diffs, [2.5, 97.5]), np.percentile(std_diffs, [2.5, 97.5])

ci_mean, ci_std = get_ci(male_ratings, female_ratings)

print(f"Mean CI is {ci_mean}")
print(f"Spread CI is {ci_std}")

# Q4: Gender Difference in Tags
tag_pvals = {}
for tag in df_tags_col:
    m_tag = gender_df[gender_df['Is_Male'] == 1][tag]
    f_tag = gender_df[gender_df['Is_Female'] == 1][tag]
    _, p = stats.ttest_ind(m_tag, f_tag)
    tag_pvals[tag] = p
sorted_tags = sorted(tag_pvals.items(), key=lambda x: x[1])  

sig_tags = [tag for tag, p in tag_pvals.items() if p < 0.05]
print(f"Significant tags ({len(sig_tags)} total): {sig_tags}")

print("3 Most Gendered Tags are ", sorted_tags[:3])
print("3 Least Gendered Tags are ", sorted_tags[-3:])

# Q5 & Q6: Difficulty Difference
m_diff = gender_df[gender_df['Is_Male'] == 1]['Avg_Difficulty_Rating']
f_diff = gender_df[gender_df['Is_Female'] == 1]['Avg_Difficulty_Rating']
_, p_val_q5 = stats.ttest_ind(m_diff, f_diff)
ci_diff_mean, _ = get_ci(m_diff, f_diff)

if p_val_q5 < 0.05:
    print("Yes, Significant difference in difficulty. CI is {ci_diff_mean}")
else:
    print("No, No significant difference in difficulty. CI is {ci_diff_mean}")


# For que Q7 to Q9. REGRESSION MODELS
def run_regression(data, target, predictors):
    X = data[predictors]
    y = data[target]
    
    scaler = StandardScaler()     # Coefficient comparison
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()
    
    # RMSE
    preds = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    # Strongest predictor (largest absolute coefficient)
    strongest_idx = np.argmax(np.abs(model.params[1:]))
    strongest_pred = predictors[strongest_idx]
    
    return model.rsquared, rmse, strongest_pred, model.summary()

# Q7: Numerical Predictors
num_preds = ['Avg_Difficulty_Rating', 'Num_Ratings', 'Has_Pepper', 'Take_Course_Again', 'Num_Online_Ratings', 'Is_Male', 'Is_Female']
r2_7, rmse_7, strong_7, summary_7 = run_regression(combined_df, 'Avg_Quality_Rating', num_preds)

print(f"(Num): R2={r2_7:.3f}, RMSE={rmse_7:.3f}, Strongest={strong_7}")


# Q8: Tag Predictors for Quality
r2_8, rmse_8, strong_8, summary_8 = run_regression(combined_df, 'Avg_Quality_Rating', df_tags_col) 

print(f"(Quality Tags): R2={r2_8:.3f}, RMSE={rmse_8:.3f}, Strongest={strong_8}")


# Q9: Tag Predictors for Difficulty
r2_9, rmse_9, strong_9, summary_9 = run_regression(combined_df, 'Avg_Difficulty_Rating', df_tags_col) 

print(f"(Difficulty Tags): R2={r2_9:.3f}, RMSE={rmse_9:.3f}, Strongest={strong_9}")

# For Q10, Classification -> predict 'Has_pepper'. I am using all factors for prediction

######-> Add PCA

X_class = combined_df[num_preds + df_tags_col].drop('Has_Pepper', axis=1)
y_class = combined_df['Has_Pepper']

X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, stratify=y_class)
clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)
auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) 

print(f"AUC Score is {auc_score:.3f}")

# Extra credit: 
# Does "Hotness" vary by state?
state_hotness = combined_df.groupby('US_State')['Has_Pepper'].mean().sort_values(ascending=False).head(5)
print(state_hotness)

# Calculate the 'personality' of each major
major_tags = combined_df.groupby('Major_Field')[df_tags_col].mean()

# Find the most 'Inspirational' majors vs the 'Tough Grader' majors
top_inspirational = major_tags['Inspirational'].sort_values(ascending=False).head(5)
top_tough = major_tags['Tough_Grader'].sort_values(ascending=False).head(5)

print("--- Extra Credit: Major Personalities ---")
print("Top 5 Most Inspirational Majors are \n", top_inspirational)
print("\nTop 5 Toughest Grading Majors are\n", top_tough)

#combined_df['Avg_Difficulty_Rating'].max()

# Normalize tags by number of ratings (tags per rating). This will help us to address the issue that more-rated professors will have more total tags
tag_cols = df_tags_col
for tag in tag_cols:
    combined_df[f'{tag}_normalized'] = combined_df[tag] / combined_df['Num_Ratings']

normalized_tag_cols = [f'{tag}_normalized' for tag in tag_cols]


from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency

# Using machine learning to group professors into 5 "types" based on their teaching tags. 
# Professors naturally cluster into distinct teaching styles, not just random combinations of traits.

##############################################Clustering teaching style
# Preparing data 
X_cluster = combined_df[normalized_tag_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(3, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=N_NUMBER, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Use 5 clusters 
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=N_NUMBER, n_init=20)
combined_df['Teaching_Archetype'] = kmeans.fit_predict(X_scaled) 


##############################################Teaching styles lead to higher ratings and which don't
archetype_profiles = {}
for cluster_id in range(optimal_k):
    cluster_data = combined_df[combined_df['Teaching_Archetype'] == cluster_id]
    
    profile = cluster_data[normalized_tag_cols].mean()
    
    # Find top 3 defining characteristics
    top_3_tags = profile.nlargest(3)
    top_3_names = [tag.replace('_normalized', '').replace('_', ' ') for tag in top_3_tags.index]
    
    avg_quality = cluster_data['Avg_Quality_Rating'].mean()
    avg_difficulty = cluster_data['Avg_Difficulty_Rating'].mean()
    pepper_rate = cluster_data['Has_Pepper'].mean()
    
    archetype_profiles[cluster_id] = {
        'top_tags': top_3_names,
        'avg_quality': avg_quality,
        'avg_difficulty': avg_difficulty,
        'pepper_rate': pepper_rate,
        'count': len(cluster_data)
    }
    
    print(f"Archetype {cluster_id + 1}: 'The {', '.join(top_3_names[:2])} Professor'")
    print(f"  Defining traits: {', '.join(top_3_names)}")
    print(f"  Avg Quality Rating: {avg_quality:.2f}")
    print(f"  Avg Difficulty: {avg_difficulty:.2f}")
    print(f"  'Hotness' Rate: {pepper_rate:.1%}")
    print(f"  N = {len(cluster_data):,} professors")
    print()

############################################## Evaluating teaching styles popularity in certain US regions
# Created regional groupings to map with provided
region_mapping = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Southeast': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK'],
    'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'Southwest': ['AZ', 'NM', 'TX'],
    'West': ['CO', 'ID', 'MT', 'NV', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
}

state_to_region = {}
for region, states in region_mapping.items():
    for state in states:
        state_to_region[state] = region

combined_df['Region'] = combined_df['US_State'].map(state_to_region)

# Chi-square test for independence between region and archetype
contingency_table = pd.crosstab(combined_df['Region'], combined_df['Teaching_Archetype'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Test: Are teaching archetypes independent of region?")
print(f"  χ² = {chi2:.2f}, p-value = {p_value:.2e}")
if p_value < 0.005:
    print("  Result: SIGNIFICANT regional variation in teaching styles detected!")
else:
    print("  Result: No significant regional variation detected.")

print("\n  Most common archetype by region:")
for region in region_mapping.keys():
    region_data = combined_df[combined_df['Region'] == region]
    if len(region_data) > 0:
        dominant_archetype = region_data['Teaching_Archetype'].mode()[0]
        proportion = (region_data['Teaching_Archetype'] == dominant_archetype).mean()
        archetype_name = ', '.join(archetype_profiles[dominant_archetype]['top_tags'][:2])
        print(f"    {region}: Archetype {dominant_archetype + 1} ({archetype_name}) - {proportion:.1%}")

############################################## Evaluation teching style with respect to Major
top_majors = combined_df['Major_Field'].value_counts().head(10).index

print("Archetype distribution in top 10 most common majors:\n")
for major in top_majors:
    major_data = combined_df[combined_df['Major_Field'] == major]
    archetype_dist = major_data['Teaching_Archetype'].value_counts(normalize=True).sort_index()
    
    # Find dominant archetype
    dominant = archetype_dist.idxmax()
    dominant_pct = archetype_dist.max()
    
    print(f"{major}:")
    print(f"  Dominant: Archetype {dominant + 1} ({dominant_pct:.1%})")
    print(f"  Distribution: {dict(zip(archetype_dist.index + 1, archetype_dist.values))}")
    print()

############################################## Evaluating teching style for easy and tough subject
combined_df['Difficulty_Level'] = pd.cut(combined_df['Avg_Difficulty_Rating'], 
                                          bins=[0, 2.5, 3.5, 5], 
                                          labels=['Easy', 'Medium', 'Hard'])

print("Average quality rating by archetype and difficulty level is")
print("(This reveals which teaching styles work best at different difficulty levels)\n")

pivot_table = combined_df.pivot_table(
    values='Avg_Quality_Rating',
    index='Teaching_Archetype',
    columns='Difficulty_Level',
    aggfunc='mean',
    observed=False
)

for archetype_id in range(optimal_k):
    archetype_name = ', '.join(archetype_profiles[archetype_id]['top_tags'][:2])
    print(f"Archetype {archetype_id + 1} ({archetype_name}):")
    for diff_level in ['Easy', 'Medium', 'Hard']:
        if diff_level in pivot_table.columns:
            rating = pivot_table.loc[archetype_id, diff_level]
            print(f"  {diff_level} courses: {rating:.2f}")
    
    if 'Hard' in pivot_table.columns and 'Easy' in pivot_table.columns:
        resilience = pivot_table.loc[archetype_id, 'Hard'] - pivot_table.loc[archetype_id, 'Easy']
        print(f"  Difficulty resilience: {resilience:+.2f} (hard vs easy)")
    print()

############################################## Evaluating teching style with gender
gender_data = combined_df[((combined_df['Is_Male'] == 1) | (combined_df['Is_Female'] == 1)) & 
                          ~((combined_df['Is_Male'] == 1) & (combined_df['Is_Female'] == 1))].copy()

for archetype_id in range(optimal_k):
    archetype_subset = gender_data[gender_data['Teaching_Archetype'] == archetype_id]
    male_pct = (archetype_subset['Is_Male'] == 1).mean()
    female_pct = (archetype_subset['Is_Female'] == 1).mean()
    
    archetype_name = ', '.join(archetype_profiles[archetype_id]['top_tags'][:2])
    print(f"Archetype {archetype_id + 1} ({archetype_name}):")
    print(f"  Male: {male_pct:.1%}, Female: {female_pct:.1%}")
    print()

# Chi-square test for gender independence from archetype
gender_archetype_table = pd.crosstab(
    gender_data['Teaching_Archetype'],
    gender_data['Is_Male']
)
chi2_gender, p_gender, _, _ = chi2_contingency(gender_archetype_table)

print(f"Chi-Square Test: Is teaching archetype independent of gender?")
print(f"  χ² = {chi2_gender:.2f}, p-value = {p_gender:.2e}")
if p_gender < 0.005:
    print("  Result: SIGNIFICANT association between gender and teaching archetype!")
else:
    print("  Result: No significant association detected.")

# 




