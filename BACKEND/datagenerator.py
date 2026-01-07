import pandas as pd
import numpy as np

np.random.seed(42)

# Load real data
df = pd.read_csv("retail_store_inventory.csv")
df['Date'] = pd.to_datetime(df['Date'])

synthetic_rows = []

# Base demand per product
product_base = df.groupby('Product ID')['Units Sold'].mean().to_dict()

for _, row in df.iterrows():
    base = product_base[row['Product ID']]

    # Price elasticity (negative)
    price_effect = np.exp(-0.08 * (row['Price'] - df['Price'].mean()))

    # Promotion uplift
    promo_effect = 1.4 if row['Holiday/Promotion'] == 'Yes' else 1.0

    # Seasonality effect
    month = row['Date'].month
    season_effect = 1.2 if month in [10, 11, 12] else 0.95

    # Controlled noise
    noise = np.random.normal(1.0, 0.15)

    units_sold = max(
        1,
        int(base * price_effect * promo_effect * season_effect * noise)
    )

    new_row = row.copy()
    new_row['Units Sold'] = units_sold
    synthetic_rows.append(new_row)

synthetic_df = pd.DataFrame(synthetic_rows)

# Augment dataset (2× size)
augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

augmented_df.to_csv("retail_store_inventory_augmented.csv", index=False)

print("Synthetic augmented dataset created!")
print("Original size:", len(df))
print("Augmented size:", len(augmented_df))
