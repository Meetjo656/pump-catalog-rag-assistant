import pandas as pd
import re

df = pd.read_csv(r"C:\Users\meetj\OneDrive\Desktop\RAG_Pumps_Chatbot\data\DOMESTIC.csv")

print("Extracted Pump Information:")
print("=" * 80)

excluded_keywords = ["SETS", "SYSTEM", "SERIES", "BOOSTER", "PRESSURE"]

# Open file for writing
with open(r"C:\Users\meetj\OneDrive\Desktop\RAG_Pumps_Chatbot\data\pump_descriptions.txt", "w") as file:
    
    for idx, row in df.iterrows():
        col_0_value = row.iloc[0]
        
        if pd.notna(col_0_value) and isinstance(col_0_value, str):
            col_0_str = col_0_value.strip()
            
            # Apply filters
            has_excluded_keyword = any(keyword in col_0_str.upper() for keyword in excluded_keywords)
            has_numbers = bool(re.search(r'\d', col_0_str))
            is_too_long = len(col_0_str) > 40
            is_model_header = col_0_str.upper() == "MODEL"
            
            if (not has_excluded_keyword and 
                has_numbers and 
                not is_too_long and 
                not is_model_header):
                
                # === Pattern-based extraction ===
                model_name = col_0_str
                kw_value = None
                hp_value = None
                size_value = None
                discharge_values = []
                
                # Scan all columns for patterns
                for value in row.iloc[1:]:
                    if pd.notna(value):
                        value_str = str(value).strip()
                        
                        # Pattern 1: Size (contains 'x' or '×' with numbers)
                        if re.search(r'\d+\s*[x×]\s*\d+', value_str, re.IGNORECASE):
                            if not size_value:
                                size_value = value_str
                        
                        # Pattern 2: Try to extract numeric discharge values
                        elif value != "-":
                            try:
                                numeric_value = float(value)
                                if numeric_value < 20:
                                    if not kw_value:
                                        kw_value = value_str
                                    elif not hp_value:
                                        hp_value = value_str
                                else:
                                    discharge_values.append(numeric_value)
                            except (ValueError, TypeError):
                                pass
                
                # Build description paragraph
                kw_text = f"{kw_value} kW" if kw_value else "power not specified"
                hp_text = f"({hp_value} HP)" if hp_value else ""
                size_text = f"with a size of {size_value} mm" if size_value else ""
                
                if discharge_values:
                    min_discharge = int(min(discharge_values))
                    max_discharge = int(max(discharge_values))
                    discharge_text = f"it delivers discharge values ranging from {min_discharge} to {max_discharge} LPH"
                else:
                    discharge_text = "discharge values are not specified"
                
                # Write paragraph to file
                paragraph = f"Pump model {model_name} has a power rating of {kw_text} {hp_text} {size_text}. Under standard test conditions, {discharge_text}. \nSource: Domestic Pump Catalog\n"
                file.write(f"{paragraph}")
                
            

print("Descriptions written to pump_descriptions.txt")