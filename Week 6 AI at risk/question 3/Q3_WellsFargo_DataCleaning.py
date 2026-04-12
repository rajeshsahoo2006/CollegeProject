"""
BADM 566 - Homework 6, Q3
Data Cleaning: WellsFargo Unstructured Data
---------------------------------------------
Clean the 'messageText' column and place results in 'CleanedText':
  - Remove Hyperlinks
  - Remove HTML tags
  - Remove Foreign Language Characters
  - Remove Spaces and Whitespaces
"""

import re
import warnings
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import os

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Week6 - WellsFargo_Unstructured.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "Week6 - WellsFargo_Cleaned.xlsx")

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading WellsFargo unstructured data...")
df = pd.read_excel(INPUT_FILE, sheet_name="wellsfargo")
print(f"Loaded {len(df)} rows.\n")

# ── Show sample BEFORE cleaning ───────────────────────────────────────────
print("=" * 70)
print("SAMPLE messageText BEFORE cleaning (first 3 rows):")
print("=" * 70)
for i, text in enumerate(df["messageText"].head(3)):
    print(f"\nRow {i}: {str(text)[:200]}...")

# ── Cleaning Function ─────────────────────────────────────────────────────
def clean_text(text):
    """Clean unstructured text data."""
    if pd.isna(text):
        return ""
    text = str(text)

    # Strip outer list brackets (data is stored as Python list repr)
    text = text.strip("[]")

    # 1. Remove HTML tags using BeautifulSoup (handles nested/malformed HTML)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 2. Remove Hyperlinks (http/https/www URLs and fragments)
    text = re.sub(r'https?://\S*', '', text)
    text = re.sub(r'http?://\S*', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'pic\.\w+\.\w+/\S+', '', text)
    # Remove bare domain-style URLs (e.g., cbsn.ws/2c06AFK)
    text = re.sub(r'\b\w+\.\w{2,}/\S+', '', text)
    # Remove .html/.htm file references
    text = re.sub(r'\S+\.html?\b', '', text)

    # 3. Remove Foreign Language Characters (keep only ASCII printable)
    text = re.sub(r'[^\x20-\x7E]', '', text)

    # 4. Remove extra Spaces and Whitespaces
    text = re.sub(r"['\"]", '', text)
    # Clean up stray commas, @-mentions formatting
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r'\s*,\s*$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ── Apply Cleaning ────────────────────────────────────────────────────────
print("\n\nCleaning data...")
df["CleanedText"] = df["messageText"].apply(clean_text)

# ── Show sample AFTER cleaning ────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAMPLE CleanedText AFTER cleaning (first 3 rows):")
print("=" * 70)
for i, text in enumerate(df["CleanedText"].head(3)):
    print(f"\nRow {i}: {text}")

# ── Summary Statistics ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)
print(f"Total rows:              {len(df)}")
print(f"Non-empty CleanedText:   {(df['CleanedText'] != '').sum()}")
print(f"Empty after cleaning:    {(df['CleanedText'] == '').sum()}")
avg_orig = df["messageText"].astype(str).str.len().mean()
avg_clean = df["CleanedText"].str.len().mean()
print(f"Avg length (original):   {avg_orig:.1f} chars")
print(f"Avg length (cleaned):    {avg_clean:.1f} chars")
print(f"Avg reduction:           {(1 - avg_clean/avg_orig)*100:.1f}%")

# ── Save to Excel ─────────────────────────────────────────────────────────
print(f"\nSaving cleaned data to: {OUTPUT_FILE}")
df.to_excel(OUTPUT_FILE, sheet_name="wellsfargo", index=False)
print("Done! Cleaned data saved successfully.")
