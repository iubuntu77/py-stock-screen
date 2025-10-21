#!/usr/bin/env python3
"""
Test script to check nsepython functionality
"""

print("Testing nsepython import and basic functionality...")
from nsepython import *
try:
    symbol = "SBIN"
    series = "EQ"
    start_date = "08-06-2021"
    end_date ="14-06-2021"
    print(equity_history(symbol,series,start_date,end_date))
    print("\n--- Testing equity_history for Adani Enterp ---")
    hist_data = equity_history("ADANIENT", "EQ", start_date="01-10-2024", end_date="31-10-2024")
    print(hist_data.columns)

except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n--- Test completed ---")