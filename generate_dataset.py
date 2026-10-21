
from modules.workload_generator import generate_workload_data

# Generate simulated cloud workload data
df = generate_workload_data(num_samples=5000)

# Save the dataset
output_path = "data/cloud_workload.csv"
df.to_csv(output_path, index=False)

print("SmartScale AI dataset generated successfully!")
print(f"Saved to: {output_path}")
print(f"Total records: {len(df)}")
print("\nDataset preview:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())
