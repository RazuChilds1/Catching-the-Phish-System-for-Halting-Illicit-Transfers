import pandas as pd

file_path = "complaints-2026-04-06_23_56.csv"

# reading header
preview = pd.read_csv(file_path, nrows=5)

print(preview.columns.tolist())
preview.head()

cols = [
    'Product',
    'Sub-product',
    'Issue',
    'Sub-issue',
    'Consumer complaint narrative',
    'State',
    'Date received'
]

df = pd.read_csv(
    file_path,
    usecols=cols,
    engine='python'
)

print(df.shape)
df.head()

df = df.dropna(subset=['Consumer complaint narrative'])

df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.strip()
df = df[df['Consumer complaint narrative'].str.len() > 20]

df = df.drop_duplicates(subset=['Consumer complaint narrative'])

relevant_products = [
    'Credit reporting or other personal consumer reports',
    'Debt collection',
    'Credit card',
    'Checking or savings account',
    'Money transfer, virtual currency, or money service',
    'Mortgage',
    'Student loan'
]

df = df[df['Product'].isin(relevant_products)]

relevant_issues = [
    'Fraud or scam',
    'Unauthorized transactions or other transaction problem',
    'Identity theft protection or other monitoring services',
    'Incorrect information on your report',
    'Improper use of your report',
    'Attempts to collect debt not owed',
    'False statements or representation',
    'Communication tactics',
    'Problem with a purchase shown on your statement',
    'Trouble using your card',
    'Problem with fraud alerts or security freezes'
]

df = df[df['Issue'].isin(relevant_issues)]

label_map = {
    'Fraud or scam': 'fraud',
    'Unauthorized transactions or other transaction problem': 'fraud',
    'Identity theft protection or other monitoring services': 'identity_theft',
    'Incorrect information on your report': 'credit_issue',
    'Improper use of your report': 'credit_issue',
    'Attempts to collect debt not owed': 'debt_fraud',
    'False statements or representation': 'scam',
    'Communication tactics': 'scam',
    'Problem with a purchase shown on your statement': 'transaction_issue',
    'Trouble using your card': 'transaction_issue',
    'Problem with fraud alerts or security freezes': 'security_issue'
}

df['label'] = df['Issue'].map(label_map)

df = df.dropna(subset=['label'])

df = df[['Consumer complaint narrative', 'label', 'Product', 'State', 'Issue', 'Sub-issue', 'Date received']]

print(df.shape)
print(df['label'].value_counts())
print(df['Issue'].value_counts())

df.to_parquet("cfpb_clean.parquet")
