import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load and Inspect the Data ---
# Load the CSV file into a DataFrame
df = pd.read_csv("CloudWatch_Traffic_Web_Attack.csv")

print("Initial data inspection:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
print(df.info())

# --- Step 2: Analyze Suspicious Activity Types ---
print("\n--- Analysis of Suspicious Activity ---")
# Count the occurrences of each unique value in 'observation_name'
observation_counts = df['observation_name'].value_counts()
print("\nTop 10 Most Common Observation Names:")
print(observation_counts.head(10).to_markdown(numalign="left", stralign="left"))

# Count the occurrences of each unique value in 'rule_names'
rule_counts = df['rule_names'].value_counts()
print("\nTop 10 Most Common Rule Names:")
print(rule_counts.head(10).to_markdown(numalign="left", stralign="left"))

# Plot the distribution of observation names
plt.figure(figsize=(10, 6))
sns.barplot(x=observation_counts.index, y=observation_counts.values, palette='viridis')
plt.title('Distribution of Suspicious Activities')
plt.xlabel('Observation Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('observation_distribution.png')
plt.show()

# --- Step 3: Analyze Geographical Distribution of Threats ---
print("\n--- Analysis of Geographical Distribution ---")
# Count the occurrences of each unique value in 'src_ip_country_code'
country_counts = df['src_ip_country_code'].value_counts()
print("\nTop 10 Source Countries of Suspicious Traffic:")
print(country_counts.head(10).to_markdown(numalign="left", stralign="left"))

# Plot the top 10 source countries
plt.figure(figsize=(12, 6))
sns.barplot(x=country_counts.head(10).index, y=country_counts.head(10).values, palette='coolwarm')
plt.title('Top 10 Source Countries of Suspicious Traffic')
plt.xlabel('Country Code')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('country_distribution.png')
plt.show()

# --- Step 4: Analyze Time-Based Trends ---
print("\n--- Analysis of Time-Based Trends ---")
# Convert 'creation_time' to datetime objects for analysis
df['creation_time'] = pd.to_datetime(df['creation_time'])

# Extract hour and day of the week
df['hour_of_day'] = df['creation_time'].dt.hour
df['day_of_week'] = df['creation_time'].dt.day_name()

# Count the occurrences of suspicious traffic by hour of the day
hourly_counts = df['hour_of_day'].value_counts().sort_index()
print("\nSuspicious Traffic by Hour of Day:")
print(hourly_counts.to_markdown(numalign="left", stralign="left"))

# Plot the hourly distribution of suspicious traffic
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o')
plt.title('Suspicious Traffic by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig('hourly_traffic_trend.png')
plt.show()

# Count the occurrences of suspicious traffic by day of the week
daily_counts = df['day_of_week'].value_counts()
# Ensure consistent order for days of the week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_counts = daily_counts.reindex(day_order)
print("\nSuspicious Traffic by Day of Week:")
print(daily_counts.to_markdown(numalign="left", stralign="left"))

# Plot the daily distribution of suspicious traffic
plt.figure(figsize=(10, 6))
sns.barplot(x=daily_counts.index, y=daily_counts.values, palette='plasma')
plt.title('Suspicious Traffic by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_traffic_trend.png')
plt.show()