import pandas as pd
import re

df = pd.read_csv(r"C:\Users\meetj\OneDrive\Desktop\RAG_Pumps_Chatbot\data\DOMESTIC.csv")

print("Valid pump model rows:")
print("=" * 80)

for idx, row in df.iterrows():
    col_0_value = row.iloc[0]
    
    if pd.notna(col_0_value) and isinstance(col_0_value, str):
        col_0_str = col_0_value.strip()
        
        # Exclude specific words
        if col_0_str not in ["Model", "SETS", "SERIES", "SYSTEM"]:
            has_letters = bool(re.search(r'[a-zA-Z]', col_0_str))
            has_numbers = bool(re.search(r'\d', col_0_str))
            
            if has_letters and has_numbers:
                print(f"Row {idx}: {col_0_str}")