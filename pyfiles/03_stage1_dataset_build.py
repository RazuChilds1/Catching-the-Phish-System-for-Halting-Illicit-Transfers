import pandas as pd
import numpy as np
df = pd.read_parquet("cfpb_clean.parquet")
print(df.shape)
print(df.head())

cols_needed = [
    'Consumer complaint narrative',
    'Product',
    'State',
    'Issue',
    'Sub-issue',
    'Date received'
]

df = df[cols_needed].copy()
print(df.shape)

#Basic cleaning safeguard
df = df.dropna(subset=['Consumer complaint narrative', 'Issue', 'Product'])
df = df[df['Consumer complaint narrative'].str.strip() != ""]
df = df.drop_duplicates(subset=['Consumer complaint narrative'])


#Standardize text columns
df['Issue'] = df['Issue'].fillna('').str.strip()
df['Sub-issue'] = df['Sub-issue'].fillna('').str.strip()
df['Product'] = df['Product'].fillna('').str.strip()
df['State'] = df['State'].fillna('UNKNOWN').str.strip()

#Create a combined issue text field for rule-based labeling
df['issue_text'] = (
    df['Issue'].str.lower() + " " + df['Sub-issue'].str.lower()
).str.strip()

df[['Issue', 'Sub-issue', 'issue_text']].head()

#Define fraud-related patterns
fraud_keywords = [
    'fraud',
    'scam',
    'identity theft',
    'stolen identity',
    'unauthorized transaction',
    'unauthorized transactions',
    'unauthorized charge',
    'unauthorized charges',
    'debt was result of identity theft',
    'security freeze',
    'fraud alert',
    'false statements or representation',
    'communication tactics'
]

#Define clearly non-fraud patterns
nonfraud_keywords = [
    'incorrect information on your report',
    'improper use of your report',
    'account status incorrect',
    'managing an account',
    'closing an account',
    'trouble during payment process',
    'problem with a purchase shown on your statement',
    'trouble using your card'
]

#Create fraud and non-fraud flags
fraud_pattern = '|'.join(fraud_keywords)
nonfraud_pattern = '|'.join(nonfraud_keywords)

df['fraud_flag'] = df['issue_text'].str.contains(fraud_pattern, case=False, regex=True)
df['nonfraud_flag'] = df['issue_text'].str.contains(nonfraud_pattern, case=False, regex=True)

print(df[['fraud_flag', 'nonfraud_flag']].mean())

#Assign Stage 1 labels carefully
df['stage1_label'] = np.where(
    (df['fraud_flag'] == True) & (df['nonfraud_flag'] == False), 1,
    np.where(
        (df['fraud_flag'] == False) & (df['nonfraud_flag'] == True), 0,
        np.nan
    )
)

print(df['stage1_label'].value_counts(dropna=False))

#Keep only clearly labeled rows
stage1_df = df.dropna(subset=['stage1_label']).copy()
stage1_df['stage1_label'] = stage1_df['stage1_label'].astype(int)

print(stage1_df.shape)
print(stage1_df['stage1_label'].value_counts())

#Rename target to something clearer
stage1_df = stage1_df.rename(columns={'stage1_label': 'is_fraud'})
print(stage1_df['is_fraud'].value_counts())

#Check class balance
print(stage1_df['is_fraud'].value_counts())
print(stage1_df['is_fraud'].value_counts(normalize=True).round(4))

#Inspect examples from both classes
stage1_df[stage1_df['is_fraud'] == 1][
    ['Issue', 'Sub-issue', 'Consumer complaint narrative']
].sample(5, random_state=42)
stage1_df[stage1_df['is_fraud'] == 0][
    ['Issue', 'Sub-issue', 'Consumer complaint narrative']
].sample(5, random_state=42)

#Create modeling dataset without leakage columns
stage1_model_df = stage1_df[
    ['Consumer complaint narrative', 'Product', 'State', 'Date received', 'is_fraud']
].copy()

print(stage1_model_df.shape)
stage1_model_df.head()

#Add simple text length feature
stage1_model_df['text_length'] = stage1_model_df['Consumer complaint narrative'].str.len()
stage1_model_df['text_length'].describe()

#Parse dates
stage1_model_df['Date received'] = pd.to_datetime(
    stage1_model_df['Date received'],
    format='%m/%d/%y',
    errors='coerce'
)

stage1_model_df['year'] = stage1_model_df['Date received'].dt.year
stage1_model_df['month'] = stage1_model_df['Date received'].dt.month

#Final missing-value cleanup
stage1_model_df = stage1_model_df.dropna(subset=['Date received'])
print(stage1_model_df.isna().sum())
print(stage1_model_df.shape)

#Final target check
print(stage1_model_df['is_fraud'].value_counts())
print(stage1_model_df['is_fraud'].value_counts(normalize=True).round(4))

stage1_model_df.to_parquet("cfpb_stage1.parquet", index=False)
print("Saved: cfpb_stage1.parquet")