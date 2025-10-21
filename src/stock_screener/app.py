"""
Streamlit frontend for Stock Screener
"""
import streamlit as st
import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_screener.screener import StockScreener
import time

# Page configuration
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide"
)

# Initialize screener
@st.cache_resource
def get_screener():
    return StockScreener()

def main():
    st.title("📈 Stock Screener")
    st.markdown("Screen stocks from NSE using various technical strategies")
    
    screener = get_screener()
    
    # Sidebar for controls
    st.sidebar.header("Screening Parameters")
    
    # Universe selection
    universes = screener.get_available_universes()
    selected_universe = st.sidebar.selectbox(
        "Select Stock Universe:",
        universes,
        help="Choose the stock universe to screen from"
    )
    
    # Strategy selection
    strategies = screener.get_available_strategies()
    strategy_options = {key: info['name'] for key, info in strategies.items()}
    
    selected_strategy_key = st.sidebar.selectbox(
        "Select Strategy:",
        list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        help="Choose the screening strategy to apply"
    )
    
    # Display strategy description
    if selected_strategy_key:
        strategy_info = strategies[selected_strategy_key]
        st.sidebar.info(f"**Strategy:** {strategy_info['description']}")
    
    # Screen button
    if st.sidebar.button("🔍 Screen Stocks", type="primary"):
        with st.spinner("Screening stocks... This may take a few minutes."):
            results = screener.screen_stocks(selected_universe, selected_strategy_key)
        
        # Display results
        if results['success']:
            st.success(f"✅ Screening completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Universe", results['universe'].upper())
            with col2:
                st.metric("Total Stocks Analyzed", results['total_stocks'])
            with col3:
                st.metric("Stocks Passing Strategy", len(results['passing_stocks']))
            
            st.subheader("📊 Results")
            
            if results['passing_stocks']:
                st.write("**Stocks that passed the screening criteria:**")
                
                # Display as columns for better layout
                cols = st.columns(3)
                for i, stock in enumerate(results['passing_stocks']):
                    with cols[i % 3]:
                        st.success(f"**{stock}**")
                
                # Download option
                stocks_text = "\n".join(results['passing_stocks'])
                st.download_button(
                    label="📥 Download Results",
                    data=stocks_text,
                    file_name=f"screened_stocks_{selected_universe}_{selected_strategy_key}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No stocks passed the screening criteria.")
                
        else:
            st.error(f"❌ Error: {results['error']}")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Available Universes:**
        - **NIFTY 50**: Top 50 companies by market cap
        - **NIFTY 500**: Top 500 companies by market cap  
        - **F&O**: Stocks available for Futures & Options trading
        """)
    
    with col2:
        st.markdown("""
        **Current Strategies:**
        - **Above VWAP**: Stocks trading above Volume Weighted Average Price
        - **Near 50-day EMA**: Stocks within 5% of 50-day Exponential Moving Average
        """)

if __name__ == "__main__":
    main()