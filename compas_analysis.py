# Converted from notebook: compas_analysis.ipynb

#!/usr/bin/env python
# coding: utf-8

# # Compas Analysis

# In[38]:


get_ipython().system('pip install pandas numpy matplotlib seaborn statsmodels -q')

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

sns.set(style="whitegrid")


# In[39]:


raw_data = pd.read_csv("dataset/compas-scores-two-years.csv")
print(f"Total rows number: {len(raw_data)}")


# In[40]:


df = raw_data.loc[:, [
    'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
    'priors_count', 'days_b_screening_arrest', 'decile_score',
    'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out',
    'compas_screening_date'    # ← add this
]]


# In[41]:


df['screening_date'] = pd.to_datetime(df['compas_screening_date'])
cutoff = pd.Timestamp('2014-04-01')


# In[42]:


df = df[df['screening_date'] <= cutoff].copy()
print(f"Rows after two-year cutoff: {len(df)}")


# In[43]:


# We filter out rows for several reasons: (similar to the Propublica study)
# 1. Screening date not within ±30 days of arrest  
# 2. Missing recidivism flag (`is_recid == -1`)  
# 3. Non-jailable offenses (`c_charge_degree == 'O'`)  
# 4. Missing COMPAS score text (`score_text == 'N/A'`)  
# 5. Only individuals with either two-year recidivism or ≥2 years out of jail


# In[44]:


df = df[
    df['days_b_screening_arrest'].between(-30, 30) &
    (df['is_recid'] != -1) &
    (df['c_charge_degree'] != 'O') &
    (df['score_text'] != 'N/A')
].copy()

print(f"Rows after filtering: {len(df)}")


# In[45]:


#length of jail stay and its correlation with COMPAS decile score.

df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

corr = df[['length_of_stay', 'decile_score']].corr().iloc[0,1]
print(f"Correlation between length_of_stay and decile_score: {corr:.4f}")


# After filtering we have the following demographic breakdown:

# In[46]:


print(df['age_cat'].value_counts(dropna=False))


# In[47]:


print(df['race'].value_counts(dropna=False))


# In[48]:


total = len(df)
for race, count in df['race'].value_counts().items():
    print(f"{race}: {count/total*100:.2f}%")


# In[49]:


print(df['score_text'].value_counts())


# In[50]:


print(pd.crosstab(df['sex'], df['race']))


# In[51]:


sex_counts = df['sex'].value_counts()
for sex, count in sex_counts.items():
    print(f"{sex}: {count/total*100:.2f}%")


# In[52]:


recid_count = df['two_year_recid'].sum()
print(f"\nNumber of recidivists: {recid_count}")
print(f"Recidivism rate: {recid_count/total*100:.2f}%")


# In[53]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

sns.countplot(x='decile_score', data=df[df['race']=="African-American"],
              order=sorted(df['decile_score'].unique()),
              ax=axes[0])
axes[0].set_title("Black Defendants' Decile Scores")
axes[0].set_ylim(0, 650)

sns.countplot(x='decile_score', data=df[df['race']=="Caucasian"],
              order=sorted(df['decile_score'].unique()),
              ax=axes[1])
axes[1].set_title("White Defendants' Decile Scores")
axes[1].set_ylim(0, 650)

for ax in axes:
    ax.set_xlabel("Decile Score")
    ax.set_ylabel("Count")

plt.tight_layout()


# In[54]:


pd.crosstab(df['decile_score'], df['race'])


# ## Racial Bias in Compas
# 
# We run a logistic regression predicting High vs. Low COMPAS score (`score_text != 'Low'`) as a function of gender, age category, race, number of priors, charge degree, and recidivism.

# In[55]:


df['high_score'] = (df['score_text'] != 'Low').astype(int)


# In[56]:


df['race_cat'] = pd.Categorical(
    df['race'],
    categories=[
        'Caucasian',
        'African-American',
        'Asian',
        'Hispanic',
        'Native American',
        'Other'
    ],
    ordered=False
)

model = smf.logit(
    formula=(
        "high_score ~ "
        "C(sex) + "
        "C(age_cat) + "
        "C(race_cat, Treatment(reference='Caucasian')) + "
        "priors_count + "
        "C(c_charge_degree) + "
        "two_year_recid"
    ),
    data=df
).fit(disp=False)


# In[57]:


# Extract the corrected coefficients:
intercept    = model.params['Intercept']
beta_black   = model.params["C(race_cat, Treatment(reference='Caucasian'))[T.African-American]"]
beta_male    = model.params['C(sex)[T.Male]']
beta_under25 = model.params['C(age_cat)[T.Less than 25]']

# Baseline probability for the reference group (Caucasian female, age 25–45, 0 priors, no recid)
p0 = np.exp(intercept) / (1 + np.exp(intercept))
print(f"Baseline P(high_score): {p0:.3f}")

# Adjusted probability ratios
def adj_ratio(beta):
    num = np.exp(beta)
    return num / (1 - p0 + p0 * num)

print(f"African-American vs. White ratio: {adj_ratio(beta_black):.3f}")
print(f"Male vs. Female ratio:           {adj_ratio(beta_male):.3f}")
print(f"<25 vs. 25–45 ratio:             {adj_ratio(beta_under25):.3f}")


# - **African-American vs. White**: Black defendants are **1.43×** as likely as White defendants to receive a High score (≈ 42.5 % higher probability).  
# - **Male vs. Female**: Male defendants are **0.84×** as likely as female defendants to get a High score (≈ 16.3 % lower probability).  
# - **Under 25 vs. 25–45**: Defendants under 25 are **2.35×** as likely as 25–45-year-olds to be classified High Risk (≈ 134.7 % higher probability).  
# 

# # Chapter 2: Exploratory Data Analysis (EDA)
# 

# ## 1. Summary Statistics

# In[60]:


# Compute descriptive statistics for selected numeric columns using quantiles
num_cols = ['age', 'priors_count', 'decile_score', 'length_of_stay']
summary_stats = pd.DataFrame({
    'mean': df[num_cols].mean(),
    'median': df[num_cols].median(),
    'std': df[num_cols].std(),
    'min': df[num_cols].min(),
    '25%': df[num_cols].quantile(0.25),
    '50%': df[num_cols].quantile(0.5),
    '75%': df[num_cols].quantile(0.75),
    'max': df[num_cols].max()
}).T
summary_stats


# ## 2. Distribution Plots

# In[61]:


import os

os.makedirs("figures", exist_ok=True)

# Histogram for age
plt.figure(figsize=(6,4))
plt.hist(df['age'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.tight_layout()
plt.savefig('figures/age_hist.png')
plt.show()

# %%
# Histogram for priors_count
plt.figure(figsize=(6,4))
plt.hist(df['priors_count'], bins=30, color='salmon', edgecolor='black')
plt.xlabel('Priors Count')
plt.ylabel('Count')
plt.title('Distribution of Priors Count')
plt.tight_layout()
plt.savefig('figures/priors_count_hist.png')
plt.show()


# ## 3. Categorical Counts

# In[63]:


# Countplot for score_text overall
plt.figure(figsize=(5,4))
sns.countplot(x='score_text', data=df, order=['Low', 'Medium', 'High'])
plt.xlabel('Score Text')
plt.ylabel('Count')
plt.title('Count of Score Text')
plt.tight_layout()
plt.savefig('figures/score_text_count.png')
plt.show()

# %%
# Countplot for score_text by race
plt.figure(figsize=(8,5))
sns.countplot(x='score_text', hue='race', data=df, order=['Low', 'Medium', 'High'])
plt.xlabel('Score Text')
plt.ylabel('Count')
plt.title('Score Text by Race')
plt.tight_layout()
plt.savefig('figures/score_text_by_race.png')
plt.show()

# %%
# Countplot for score_text by sex
plt.figure(figsize=(6,4))
sns.countplot(x='score_text', hue='sex', data=df, order=['Low', 'Medium', 'High'])
plt.xlabel('Score Text')
plt.ylabel('Count')
plt.title('Score Text by Sex')
plt.tight_layout()
plt.savefig('figures/score_text_by_sex.png')
plt.show()


# ## 4. Grouped Recidivism Rates

# In[66]:


# Recidivism rate by race
recid_by_race = df.groupby('race')['two_year_recid'].mean().sort_values(ascending=False)
print("Recidivism Rate by Race:")
print(recid_by_race)

# %%
# Recidivism rate by age_cat
recid_by_age = df.groupby('age_cat')['two_year_recid'].mean().sort_values(ascending=False)
print("Recidivism Rate by Age Category:")
print(recid_by_age)

# %%
# Recidivism rate by c_charge_degree
recid_by_charge = df.groupby('c_charge_degree')['two_year_recid'].mean().sort_values(ascending=False)
print("Recidivism Rate by Charge Degree:")
print(recid_by_charge)


# # Chapter 3: Predictive Modeling and Fairness Evaluation

# ## 1. Train/Test Split

# In[67]:


import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Create directories if they don't exist
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Get one-hot encodings for categorical features
categorical_features = pd.get_dummies(df[['race', 'sex', 'c_charge_degree']], drop_first=False)

# Create feature matrix
X = pd.concat([
    df[['decile_score', 'priors_count', 'age']],
    categorical_features
], axis=1)

# Define target variable
y = df['two_year_recid']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# ## 2. Model Training

# In[68]:


# Create and fit logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[68b]:


# Create and fit XGBoost model
print("\nTraining XGBoost model...")
# Added use_label_encoder=False to avoid a warning with recent xgboost versions
# Added eval_metric to specify for potential future use or suppress warnings
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
print("XGBoost model trained.")


# ## 3. Performance Metrics

# In[69]:


# Predict on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")


# In[69b]:


# Predict on test set with XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Print classification report for XGBoost
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# Compute ROC AUC for XGBoost
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"ROC AUC (XGBoost): {roc_auc_xgb:.4f}")


# ## 4. ROC Curve

# In[70]:


# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('figures/roc_curve.png')
plt.show()


# In[70b]:


# Compute ROC curve points for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

# Plot ROC curve for XGBoost
plt.figure(figsize=(6, 6))
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label=f'XGBoost ROC curve (AUC = {roc_auc_xgb:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
plt.legend(loc="lower right")
plt.savefig('figures/roc_curve_xgb.png')
plt.show()


# ## 5. Fairness Metrics by Race

# In[71]:


# Create a DataFrame to store fairness metrics
fairness_metrics = []

# Get race values from test data (need to reconstruct from one-hot columns)
race_columns = [col for col in X_test.columns if col.startswith('race_')]
test_indices = X_test.index

# Recreate race column for test set
test_with_race = pd.DataFrame({'race': df.loc[test_indices, 'race']})
test_with_race['y_true'] = y_test.values
test_with_race['y_pred'] = y_pred

# Calculate metrics for each race
for race in test_with_race['race'].unique():
    race_mask = test_with_race['race'] == race

    # Get race-specific predictions and labels
    y_true_race = test_with_race.loc[race_mask, 'y_true']
    y_pred_race = test_with_race.loc[race_mask, 'y_pred']

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_race, y_pred_race).ravel()

    # Calculate FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    # Add to results
    fairness_metrics.append({
        'race': race,
        'count': len(y_true_race),
        'FPR': fpr,
        'FNR': fnr
    })

# Create DataFrame and display results
fairness_df = pd.DataFrame(fairness_metrics)
fairness_df = fairness_df.sort_values('count', ascending=False)
print("Fairness Metrics by Race:")
print(fairness_df)


# In[71b]:


# Create a DataFrame to store fairness metrics for XGBoost
fairness_metrics_xgb = []

# X_test.index (as test_indices) and y_test are available from the logistic regression section (In[71])
# df is globally available

# Recreate race column for test set, specific for XGBoost predictions
test_with_race_xgb = pd.DataFrame({'race': df.loc[X_test.index, 'race']})
test_with_race_xgb['y_true'] = y_test.values # y_test aligns with X_test from train_test_split
test_with_race_xgb['y_pred_xgb'] = y_pred_xgb # y_pred_xgb is from xgb_model.predict(X_test)

# Calculate metrics for each race for XGBoost
print("\nFairness Metrics by Race (XGBoost):")
for race in test_with_race_xgb['race'].unique():
    race_mask_xgb = test_with_race_xgb['race'] == race

    y_true_race_xgb = test_with_race_xgb.loc[race_mask_xgb, 'y_true']
    y_pred_race_xgb = test_with_race_xgb.loc[race_mask_xgb, 'y_pred_xgb']

    if len(y_true_race_xgb) > 0: # Ensure there are samples for this race
        # Compute confusion matrix for XGBoost, ensure all 4 values are returned
        cm_xgb = confusion_matrix(y_true_race_xgb, y_pred_race_xgb, labels=[0, 1])
        tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

        # Calculate FPR and FNR for XGBoost
        fpr_val_xgb = fp_xgb / (fp_xgb + tn_xgb) if (fp_xgb + tn_xgb) > 0 else np.nan
        fnr_val_xgb = fn_xgb / (fn_xgb + tp_xgb) if (fn_xgb + tp_xgb) > 0 else np.nan
    else: # Handle cases where a race subgroup might be empty in the test set after filtering
        fpr_val_xgb, fnr_val_xgb = np.nan, np.nan

    # Add to results for XGBoost
    fairness_metrics_xgb.append({
        'race': race,
        'count': len(y_true_race_xgb),
        'FPR_xgb': fpr_val_xgb,
        'FNR_xgb': fnr_val_xgb
    })

# Create DataFrame and display results for XGBoost
fairness_df_xgb = pd.DataFrame(fairness_metrics_xgb)
fairness_df_xgb = fairness_df_xgb.sort_values('count', ascending=False)
print(fairness_df_xgb[['race', 'count', 'FPR_xgb', 'FNR_xgb']])


# ## 6. Save Model and Figures

# In[72]:


# Save the model to a file
joblib.dump(model, 'models/logistic_model.pkl')
print("Model saved to models/logistic_model.pkl")

# Save the XGBoost model to a file
joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("XGBoost model saved to models/xgb_model.pkl")


# # Chapter 4: Explainability Snapshot

# ## 1. Load the Model and Test Data

# In[75]:


get_ipython().system(' pip3 install shap xgboost')

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Ensure figures and models directories exist
os.makedirs("models", exist_ok=True) # For saving fallback model if needed

xgb_model = None
model_path = 'models/xgb_model.pkl'
X_test_shap = None # Use a distinct name for X_test in this chapter to avoid scope issues if re-run
y_test_shap = None # Use a distinct name for y_test

# Attempt to load the XGBoost model
try:
    xgb_model = joblib.load(model_path)
    print(f"XGBoost model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"WARNING: XGBoost model not found at {model_path}.")
    # Optional: attempt to retrain if X_train, y_train are available from Chapter 3.
    # This part is simplified; a full robust fallback would require X_train, y_train to be definitively in scope.
    if 'X_train' in locals() and 'y_train' in locals() and 'xgb' in globals():
        print("Attempting to train a fallback XGBoost model...")
        try:
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train) # Assumes X_train, y_train are from Ch3
            joblib.dump(xgb_model, model_path)
            print(f"Fallback XGBoost model trained and saved to {model_path}")
        except Exception as e_train:
            print(f"Error training fallback model: {e_train}")
    else:
        print("Cannot train fallback model: X_train, y_train, or xgboost not available.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")

# Ensure X_test and y_test are available (preferring those from Chapter 3 execution)
if 'X_test' in locals() and 'y_test' in locals():
    X_test_shap = X_test
    y_test_shap = y_test
    print("Using X_test and y_test from Chapter 3.")
else:
    print("WARNING: X_test or y_test not found in the global scope from Chapter 3.")
    print("Attempting to reconstruct X_test/y_test. This assumes 'df' DataFrame is available and processed.")
    if 'df' in locals() and 'pd' in globals() and 'train_test_split' in globals():
        try:
            categorical_features_reco = pd.get_dummies(df[['race', 'sex', 'c_charge_degree']], drop_first=False)
            X_reco = pd.concat([
                df[['decile_score', 'priors_count', 'age']],
                categorical_features_reco
            ], axis=1)
            y_reco = df['two_year_recid']
            _, X_test_shap, _, y_test_shap = train_test_split(
                X_reco, y_reco, test_size=0.3, stratify=y_reco, random_state=42
            )
            print("X_test and y_test reconstructed for SHAP chapter.")
            # If model is still None, and we just reconstructed data, try training if X_train_reco also made
            if xgb_model is None and 'X_train' not in locals() and 'xgb' in globals(): # A bit more complex fallback
                 X_train_reco, _, y_train_reco, _ = train_test_split(X_reco, y_reco, test_size=0.3, stratify=y_reco, random_state=42)
                 print("Attempting to train XGBoost model with reconstructed training data...")
                 xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                 xgb_model.fit(X_train_reco, y_train_reco)
                 joblib.dump(xgb_model, model_path)
                 print(f"Fallback XGBoost model (with reconstructed data) trained and saved to {model_path}")

        except Exception as e_reco:
            print(f"Error reconstructing X_test/y_test: {e_reco}")
    else:
        print("ERROR: Cannot reconstruct X_test/y_test: 'df', 'pd', or 'train_test_split' not available.")

if xgb_model is None or X_test_shap is None:
    print("\n### ERROR: XGBoost model or X_test_shap is not available. SHAP analysis cannot proceed. ###")
else:
    print("\nModel and X_test_shap are ready for SHAP analysis.")

# ## 2. Initialize SHAP Explainer

# In[76]:


explainer = None
shap_values = None

if xgb_model is not None and X_test_shap is not None:
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_shap) # X_test_shap must be a DataFrame or numpy array
        print("SHAP explainer initialized and shap_values computed.")
    except Exception as e_shap_init:
        print(f"Error initializing SHAP or computing values: {e_shap_init}")
else:
    print("Skipping SHAP initialization: model or X_test_shap is missing.")

# ## 3. Global Feature Importance

# In[77]:


if shap_values is not None and X_test_shap is not None:
    try:
        plt.figure() 
        shap.summary_plot(shap_values, X_test_shap, plot_type="bar", show=False)
        plt.title("Mean |SHAP| Feature Importance")
        plt.savefig('figures/shap_bar.png', bbox_inches='tight')
        plt.show()
        print("Global feature importance plot saved to figures/shap_bar.png")
    except Exception as e_summary_plot:
        print(f"Error generating SHAP summary plot: {e_summary_plot}")
else:
    print("Skipping global feature importance plot: SHAP values or X_test_shap are missing.")

# ## 4. Top-5 Features Table

# In[78]:


if shap_values is not None and X_test_shap is not None:
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = X_test_shap.columns
        
        shap_summary_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        top_5_features = shap_summary_df.sort_values(by='mean_abs_shap', ascending=False).head(5)
        
        print("\nTop-5 Features by Mean Absolute SHAP Value:")
        print(top_5_features)
    except Exception as e_top_features:
        print(f"Error calculating top-5 features: {e_top_features}")
else:
    print("Skipping top-5 features table: SHAP values or X_test_shap are missing.")

# ## 5. Single-Case Waterfall Explanation

# In[79]:


if explainer is not None and shap_values is not None and X_test_shap is not None and len(X_test_shap) > 0:
    i = 0 # Index for the test sample
    if i < len(X_test_shap):
        try:
            plt.figure()
            
            # Construct SHAP Explanation object for the instance
            shap_explanation_instance = shap.Explanation(
                values=shap_values[i],
                base_values=explainer.expected_value, # E[f(x)] for the model
                data=X_test_shap.iloc[i].values,    # Actual feature values for the instance
                feature_names=X_test_shap.columns.tolist()
            )
            
            shap.plots.waterfall(shap_explanation_instance, show=False)
            
            plt.title(f"SHAP Waterfall for Test Sample {i}")
            plt.savefig(f'figures/shap_waterfall.png', bbox_inches='tight')
            plt.show()
            print(f"SHAP waterfall plot for sample {i} saved to figures/shap_waterfall.png")
        except Exception as e_waterfall:
            print(f"Error generating SHAP waterfall plot: {e_waterfall}")
    else:
        print(f"Test sample index {i} is out of bounds for X_test_shap with length {len(X_test_shap)}.")
else:
    print("Skipping single-case waterfall plot: explainer, SHAP values, or X_test_shap are missing/empty.")

# %%
# End of Chapter 4
print("\nChapter 4 (SHAP Explainability) processing complete.")

# %% [markdown]
# ## 6. Conditional SHAP Analysis (Decile Score < 5 vs. >= 5)
# This section explores how feature importance differs for individuals with a COMPAS decile score below 5 versus those with a decile score of 5 or greater.

# %%
if explainer is not None and X_test_shap is not None and 'decile_score' in X_test_shap.columns:
    print("\nStarting Conditional SHAP Analysis based on 'decile_score'...")

    # Filter X_test_shap based on decile_score
    X_test_below_5 = X_test_shap[X_test_shap['decile_score'] < 5]
    X_test_above_5 = X_test_shap[X_test_shap['decile_score'] >= 5]

    print(f"Number of samples with decile_score < 5: {len(X_test_below_5)}")
    print(f"Number of samples with decile_score >= 5: {len(X_test_above_5)}")

    if not X_test_below_5.empty:
        try:
            print("Computing SHAP values for decile_score < 5...")
            shap_values_below_5 = explainer.shap_values(X_test_below_5)
            
            plt.figure()
            shap.summary_plot(shap_values_below_5, X_test_below_5, plot_type="bar", show=False)
            plt.title("Mean |SHAP| - Decile Score < 5")
            plt.savefig('figures/shap_bar_decile_below_5.png', bbox_inches='tight')
            plt.show()
            print("Conditional SHAP summary plot for decile_score < 5 saved to figures/shap_bar_decile_below_5.png")
        except Exception as e_shap_below:
            print(f"Error during SHAP analysis for decile_score < 5: {e_shap_below}")
    else:
        print("No samples found with decile_score < 5. Skipping SHAP analysis for this subset.")

    if not X_test_above_5.empty:
        try:
            print("\nComputing SHAP values for decile_score >= 5...")
            shap_values_above_5 = explainer.shap_values(X_test_above_5)

            plt.figure()
            shap.summary_plot(shap_values_above_5, X_test_above_5, plot_type="bar", show=False)
            plt.title("Mean |SHAP| - Decile Score >= 5")
            plt.savefig('figures/shap_bar_decile_above_5.png', bbox_inches='tight')
            plt.show()
            print("Conditional SHAP summary plot for decile_score >= 5 saved to figures/shap_bar_decile_above_5.png")
        except Exception as e_shap_above:
            print(f"Error during SHAP analysis for decile_score >= 5: {e_shap_above}")
    else:
        print("No samples found with decile_score >= 5. Skipping SHAP analysis for this subset.")
        
else:
    print("\nSkipping conditional SHAP analysis: explainer, X_test_shap, or 'decile_score' column is missing.")

# %%
# This empty In[] cell is just to mimic notebook structure if converted back.
# It doesn't affect python script execution.

