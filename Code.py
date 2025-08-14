 # Import necessary libraries
import pandas as pd
import numpy as np
 # Load the dataset
df = pd.read_csv(r'C:\Users\sheka\OneDrive\Desktop\Electric Vehicle Sales by State in India.csv')
 # Display the first few rows of the dataset
print(df.head())
import pandas as pd

# Read the CSV file into a pandas DataFrame.
df = pd.read_csv(r'C:\Users\sheka\OneDrive\Desktop\Electric Vehicle Sales by State in India.csv')

# Display the first 5 rows of the DataFrame.
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display the information about the DataFrame.
print(df.info())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a pandas DataFrame.
df = pd.read_csv(r'C:\Users\sheka\OneDrive\Desktop\Electric Vehicle Sales by State in India.csv')

# Convert `Year` to integer type.
df['Year'] = df['Year'].astype('int')

# Convert `Date` to datetime objects.
df['Date'] = pd.to_datetime(df['Date'])

# Drop the `Month_Name` column as it is redundant.
df = df.drop(columns=['Month_Name'])

# Display the information about the DataFrame to verify the changes.
print(df.info())

# Aggregate EV sales quantity by `Vehicle_Category`.
sales_by_category = df.groupby('Vehicle_Category')['EV_Sales_Quantity'].sum().reset_index()

# Sort the values for better visualization
sales_by_category = sales_by_category.sort_values(by='EV_Sales_Quantity', ascending=False)

# Create the bar plot.
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Category', y='EV_Sales_Quantity', data=sales_by_category, palette='viridis')
plt.title('Total EV Sales by Vehicle Category')
plt.xlabel('Vehicle Category')
plt.ylabel('EV Sales Quantity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ev_sales_by_category_bar_plot.png')
plt.show()

# Display the aggregated data.
print(sales_by_category.to_markdown(index=False, numalign="left", stralign="left"))
