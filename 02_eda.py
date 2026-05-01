import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load cleaned dataset
df = pd.read_parquet("cfpb_clean.parquet")
print(df.shape)
df.head()
print(df.info())

#Missing values check
df.isna().sum().sort_values(ascending=False)

#Quick preview of label distribution
print(df['label'].value_counts())

#Plot label distribution
df['label'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()


#Product distribution
print(df['Product'].value_counts().head(15))

#Plot top products
df['Product'].value_counts().head(15).plot(kind='bar', figsize=(12, 6))
plt.title("Top 15 Products")
plt.xlabel("Product")
plt.ylabel("Count")
plt.xticks(rotation=75)
plt.tight_layout()


#Top states
print(df['State'].value_counts().head(15))

#Plot top states
df['State'].value_counts().head(15).plot(kind='bar', figsize=(10, 6))
plt.title("Top 15 States")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()


#Top issues
print(df['Issue'].value_counts().head(15))

#Plot top issues
df['Issue'].value_counts().head(15).plot(kind='bar', figsize=(12, 6))
plt.title("Top 15 Issues")
plt.xlabel("Issue")
plt.ylabel("Count")
plt.xticks(rotation=75)
plt.tight_layout()


#Create text length feature
df['text_length'] = df['Consumer complaint narrative'].str.len()
df[['Consumer complaint narrative', 'text_length']].head()
df['text_length'].describe()

df['text_length'].plot(kind='hist', bins=50, figsize=(10, 6))
plt.title("Distribution of Complaint Narrative Length")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()


df.groupby('label')['text_length'].mean().sort_values(ascending=False)      

df.groupby('label')['text_length'].mean().sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 6)
)
plt.title("Average Narrative Length by Label")
plt.xlabel("Label")
plt.ylabel("Average Number of Characters")
plt.xticks(rotation=45)
plt.tight_layout()


df.groupby('label')['text_length'].median().sort_values(ascending=False)

df['Date received'] = pd.to_datetime(
    df['Date received'],
    format='%m/%d/%y',
    errors='coerce'
)

df['year'] = df['Date received'].dt.year
df['month'] = df['Date received'].dt.to_period('M')

print(df[['Date received', 'year', 'month']].head())

df['year'].value_counts().sort_index()

df['year'].value_counts().sort_index().plot(kind='bar', figsize=(10, 6))
plt.title("Complaints by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()


monthly_counts = df['month'].value_counts().sort_index()
monthly_counts.tail(24)

monthly_counts.plot(figsize=(14, 6))
plt.title("Monthly Complaint Volume")
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()

pd.crosstab(df['label'], df['Product'])
pd.crosstab(df['label'], df['Product'], normalize='index')

print("Unique labels:", df['label'].nunique())
print("Unique products:", df['Product'].nunique())
print("Unique states:", df['State'].nunique())
print("Unique issues:", df['Issue'].nunique())
print("Unique sub-issues:", df['Sub-issue'].nunique())

df['Consumer complaint narrative'].duplicated().sum()
label_pct = df['label'].value_counts(normalize=True) * 100
print(label_pct.round(2))

label_pct.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title("Label Percentage Distribution")
plt.xlabel("Label")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.to_parquet("cfpb_eda_ready.parquet", index=False)
print("Saved: cfpb_eda_ready.parquet")

