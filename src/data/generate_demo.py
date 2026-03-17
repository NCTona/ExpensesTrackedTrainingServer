import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_realistic_expenses(num_users=20, months=12):
    """
    Generate synthetic transaction data (Demo Data) based on real user data.
    Used when real data (transactions.csv) is too sparse for AI training.

    Target: ~5.9M/month/user (based on real data analysis).
      Cat 1 (Nha o):    ~2,500,000 (1 giao dich co dinh)
      Cat 2 (An uong):  ~1,600,000 (27 giao dich, ~60k/giao dich)
      Cat 3 (Quan ao):    ~500,000 (1 giao dich)
      Cat 4 (Di lai):     ~130,000 (4 giao dich, ~32k/giao dich)
      Cat 5 (Sac dep):    ~350,000 (1 giao dich)
      Cat 6 (Giao luu):   ~150,000 (1 giao dich)
      Cat 7 (Y te):       ~200,000 (hiem khi)
      Cat 8 (Hoc tap):    ~500,000 (1 giao dich)
    """
    now = datetime.now()
    start_date = now - timedelta(days=months * 30)

    records = []
    user_ids = [1000 + i for i in range(num_users)]

    random.seed(42)
    np.random.seed(42)

    for user_id in user_ids:
        # Bien dong nhe giua cac user (0.9 - 1.1)
        user_multiplier = random.uniform(0.9, 1.1)

        for m in range(months + 1):
            current_month_date = start_date + timedelta(days=m * 30)
            if current_month_date > now:
                break

            month_str = current_month_date.strftime("%Y-%m")

            # --- CAT 1: NHA O (Co dinh ~2,500,000, 1 giao dich/thang) ---
            housing_amount = 2500000.0 * user_multiplier
            records.append({
                "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 1,
                "amount": float(round(housing_amount, -3)), "date": f"{month_str}-05",
                "note": "Housing (Demo)", "type": "expense"
            })

            # --- CAT 2: AN UONG (~1,600,000/thang, ~27 giao dich) ---
            # 1 bua/ngay, ~57k/bua
            for day in range(1, 28):
                date_str = f"{month_str}-{day:02d}"
                amount = random.choice([40000, 50000, 60000, 70000, 80000]) * user_multiplier
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 2,
                    "amount": float(round(amount, -2)), "date": date_str,
                    "note": "Food (Demo)", "type": "expense"
                })

            # --- CAT 4: DI LAI (~130,000/thang, 4 giao dich) ---
            transport_days = random.sample(range(1, 28), 4)
            for day in transport_days:
                date_str = f"{month_str}-{day:02d}"
                amount = random.choice([20000, 30000, 40000, 50000]) * user_multiplier
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 4,
                    "amount": float(round(amount, -2)), "date": date_str,
                    "note": "Transport (Demo)", "type": "expense"
                })

            # --- CAT 3: QUAN AO (~500,000, 1 giao dich/thang) ---
            if random.random() < 0.85:
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 3,
                    "amount": float(round(500000.0 * user_multiplier, -3)),
                    "date": f"{month_str}-{random.randint(10, 25):02d}",
                    "note": "Clothes (Demo)", "type": "expense"
                })

            # --- CAT 5: SAC DEP (~350,000, 1 giao dich/thang) ---
            if random.random() < 0.7:
                amount = random.choice([150000, 300000, 500000]) * user_multiplier
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 5,
                    "amount": float(round(amount, -3)),
                    "date": f"{month_str}-{random.randint(5, 20):02d}",
                    "note": "Beauty (Demo)", "type": "expense"
                })

            # --- CAT 6: GIAO LUU (~150,000, 1 giao dich/thang) ---
            if random.random() < 0.7:
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 6,
                    "amount": float(round(150000.0 * user_multiplier, -3)),
                    "date": f"{month_str}-{random.randint(1, 28):02d}",
                    "note": "Social (Demo)", "type": "expense"
                })

            # --- CAT 7: Y TE (~200,000, hiem khi ~20% thang) ---
            if random.random() < 0.2:
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 7,
                    "amount": float(round(200000.0 * user_multiplier, -3)),
                    "date": f"{month_str}-{random.randint(1, 28):02d}",
                    "note": "Medical (Demo)", "type": "expense"
                })

            # --- CAT 8: HOC TAP (~500,000, 1 giao dich/thang) ---
            if random.random() < 0.7:
                records.append({
                    "transaction_id": len(records) + 1, "user_id": user_id, "category_id": 8,
                    "amount": float(round(500000.0 * user_multiplier, -3)),
                    "date": f"{month_str}-{random.randint(1, 28):02d}",
                    "note": "Education (Demo)", "type": "expense"
                })

    return pd.DataFrame(records)

if __name__ == "__main__":
    output_path = "data/raw/transactions_demo.csv"
    os.makedirs("data/raw", exist_ok=True)
    df = generate_realistic_expenses()
    df.to_csv(output_path, index=False)
    print(f"Created {len(df)} transactions demo at {output_path}")
