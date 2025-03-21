import pandas as pd

# Load the dataset from a CSV file
df = pd.read_csv("C:\\Users\\hanvi\\Research\\melbourne\\discharge_data.csv")  # Ensure CSV has headers

# Convert the "Date" column to pandas datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", errors='coerce')

# Drop any rows where conversion failed
df = df.dropna(subset=["Date"])

# Extract only the date part (without time)
df["Only_Date"] = df["Date"].dt.date

# Select the first occurrence of each day
df_filtered = df.groupby("Only_Date").first().reset_index()

# Rename the date column back
df_filtered = df_filtered.rename(columns={"Only_Date": "Date"})

# Save the filtered data to a new CSV file
df_filtered.to_csv("filtered_data.csv", index=False)

print("Filtered data saved to filtered_data.csv")
