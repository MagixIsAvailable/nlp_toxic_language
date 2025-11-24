import google.generativeai as genai
import pandas as pd
import time
import os

# --- CONFIGURATION ---
API_KEY = ""  # Paste your key here
INPUT_FILE = "Data/live_recording_data.csv"
OUTPUT_FILE = "Data/audited_data.csv"

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def get_gemini_verdict(text):
    """Asks Gemini to act as a safety judge."""
    prompt = f"""
    Act as a content moderation expert. Classify the following text as 'Toxic' or 'Safe'.
    
    Rules:
    - 'Toxic' includes hate speech, threats, insults, and severe harassment.
    - 'Safe' includes benign statements, negative opinions (e.g. 'I hate broccoli'), and non-abusive slang.
    - Be smarter than a keyword filter. Understand context.

    Text: "{text}"

    Reply ONLY with the number:
    1 (if Toxic)
    0 (if Safe)
    """
    try:
        response = model.generate_content(prompt)
        # Clean up response (remove whitespace/newlines)
        return int(response.text.strip())
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- MAIN LOOP ---
if not os.path.exists(INPUT_FILE):
    print(f" File not found: {INPUT_FILE}")
    exit()

print(f" Starting Audit on {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Create a new column for the "Teacher's" label
df['gemini_label'] = -1 

for index, row in df.iterrows():
    text = row['text']
    current_label = row['model_label']
    
    print(f"[{index+1}/{len(df)}] Checking: '{text}'...", end=" ")
    
    # Ask Gemini
    true_label = get_gemini_verdict(text)
    
    if true_label is not None:
        df.at[index, 'gemini_label'] = true_label
        
        # specific check to see if Gemini disagreed with your model
        if true_label != current_label:
            print(f" CORRECTION! (Model: {current_label} -> Gemini: {true_label})")
        else:
            print(" Agreed.")
    
    # Sleep briefly to avoid hitting API rate limits
    time.sleep(1.0) 

# Save the "Gold Standard" dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Audit Complete. Saved to {OUTPUT_FILE}")
print("You can now use 'gemini_label' as the ground truth for fine-tuning!")