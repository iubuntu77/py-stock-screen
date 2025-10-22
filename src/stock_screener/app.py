"""
Streamlit frontend for Stock Screener
"""
import streamlit as st
import sys
import os
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
    try:
        # Initialize screener with caching enabled
        return StockScreener(enable_cache=True, cache_duration_hours=4)
    except Exception as e:
        st.error(f"Error initializing screener: {e}")
        return None

def main():
    st.title("📈 Stock Screener")
    st.markdown("Screen stocks from NSE using various technical strategies")
    
    screener = get_screener()
    if not screener:
        st.error("Failed to initialize screener. Please check the logs.")
        st.stop()
    
    try:
        strategy_manager = screener.strategy_manager
    except Exception as e:
        st.error(f"Error accessing strategy manager: {e}")
        st.stop()
    
    # Sidebar for controls
    st.sidebar.header("Screening Parameters")
    
    # Data provider info
    with st.sidebar.expander("📊 Data Provider Info"):
        provider_info = screener.get_data_provider_info()
        current_provider = provider_info.get('current_provider', 'Unknown')
        
        st.write(f"**Current:** {current_provider}")
        st.write(f"**Fallback:** {provider_info.get('fallback_provider', 'Unknown')}")
        
        # Provider status
        status = provider_info.get('provider_status', {})
        for provider, info in status.items():
            status_icon = "✅" if info.get('working', False) else "❌"
            st.write(f"{status_icon} {provider}")
        
        # Provider selection
        available_providers = provider_info.get('available_providers', [])
        if len(available_providers) > 1:
            new_provider = st.selectbox(
                "Switch Provider:",
                available_providers,
                index=available_providers.index(current_provider) if current_provider in available_providers else 0
            )
            
            if st.button("Switch") and new_provider != current_provider:
                if screener.set_data_provider(new_provider):
                    st.success(f"Switched to {new_provider}")
                    st.rerun()
                else:
                    st.error("Failed to switch provider")
    
    # Cache management
    with st.sidebar.expander("💾 Cache Management"):
        cache_info = screener.get_cache_info()
        
        if cache_info['enabled']:
            st.write(f"**Status:** ✅ Enabled")
            st.write(f"**Duration:** {cache_info['duration_hours']} hours")
            st.write(f"**Memory entries:** {cache_info['memory_entries']}")
            st.write(f"**Disk entries:** {cache_info['disk_entries']}")
            st.write(f"**Cache size:** {cache_info['total_size_mb']} MB")
            
            # Cache management buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All"):
                    screener.clear_cache()
                    st.success("Cache cleared!")
                    st.rerun()
            
            with col2:
                # Universe selection for selective cache clearing
                universes = screener.get_available_universes()
                selected_clear_universe = st.selectbox(
                    "Clear specific:",
                    [""] + universes,
                    key="clear_universe"
                )
                if selected_clear_universe and st.button("Clear"):
                    screener.clear_cache(selected_clear_universe)
                    st.success(f"Cleared {selected_clear_universe} cache!")
                    st.rerun()
        else:
            st.write("**Status:** ❌ Disabled")
    
    # Audit log management
    with st.sidebar.expander("📋 Audit Logs"):
        try:
            log_stats = screener.get_audit_log_stats()
            
            st.write(f"**Log Directory:** {log_stats['log_dir']}")
            st.write(f"**File Count:** {log_stats['file_count']}")
            st.write(f"**Total Size:** {log_stats['total_size_mb']} MB")
            st.write(f"**Current Session:** {log_stats['current_session']}")
            
            if log_stats['files']:
                st.write("**Log Files:**")
                for file_info in log_stats['files']:
                    st.write(f"• {file_info['name']} ({file_info['size_kb']} KB)")
            
            # Log file viewer
            if st.button("📖 View Latest Log"):
                st.session_state.show_logs = True
            
        except Exception as e:
            st.write(f"Error accessing logs: {e}")
    
    # Universe selection
    universes = screener.get_available_universes()
    selected_universe = st.sidebar.selectbox(
        "Select Stock Universe:",
        universes,
        help="Choose the stock universe to screen from"
    )
    
    # Strategy selection with enhanced UI
    st.sidebar.subheader("📋 Strategy Selection")
    
    # Strategy mode selection
    # Check if chaining is available
    chaining_available = True
    try:
        # Test if chaining methods are available
        strategy_manager.get_filter_strategies()
        strategy_manager.get_strategies_by_category()
    except:
        chaining_available = False
    
    if chaining_available:
        strategy_mode = st.sidebar.radio(
            "Strategy Mode:",
            ["Single Strategy", "Chain Strategies"],
            help="Choose single strategy or chain multiple strategies"
        )
    else:
        strategy_mode = "Single Strategy"
        st.sidebar.info("Strategy chaining not available - using single strategy mode")
    
    if strategy_mode == "Chain Strategies":
        # Strategy chaining interface
        st.sidebar.subheader("🔗 Strategy Chain Builder")
        
        # Initialize session state for strategy chain
        if 'strategy_chain' not in st.session_state:
            st.session_state.strategy_chain = []
        
        # Get available filter strategies (only these can be chained)
        try:
            filter_strategies = strategy_manager.get_filter_strategies()
            if not filter_strategies:
                st.sidebar.warning("No filter strategies available for chaining. Using single strategy mode.")
                strategy_mode = "Single Strategy"
        except Exception as e:
            st.sidebar.error(f"Strategy chaining not available: {e}")
            st.sidebar.info("Using single strategy mode instead.")
            strategy_mode = "Single Strategy"
            filter_strategies = {}
        
        # Add strategy to chain
        with st.sidebar.expander("➕ Add Strategy to Chain", expanded=True):
            available_strategies = {key: strategy.name for key, strategy in filter_strategies.items()}
            
            new_strategy_key = st.selectbox(
                "Select Strategy:",
                list(available_strategies.keys()),
                format_func=lambda x: available_strategies[x],
                key="new_strategy_select"
            )
            
            # Operator selection (except for first strategy)
            if len(st.session_state.strategy_chain) > 0:
                operator = st.selectbox(
                    "Logical Operator:",
                    ["AND", "OR"],
                    help="How to combine with previous strategies"
                )
            else:
                operator = None
            
            # Parameters for selected strategy
            if new_strategy_key:
                new_strategy = filter_strategies[new_strategy_key]
                param_config = new_strategy.get_parameter_config()
                strategy_parameters = {}
                
                if param_config:
                    st.write("**Parameters:**")
                    for param_name, config in param_config.items():
                        if config['type'] == 'slider':
                            value = st.slider(
                                config['label'],
                                min_value=config['min'],
                                max_value=config['max'],
                                value=config['default'],
                                step=config['step'],
                                help=config['help'],
                                key=f"chain_param_{param_name}"
                            )
                            strategy_parameters[param_name] = value
                        elif config['type'] == 'selectbox':
                            value = st.selectbox(
                                config['label'],
                                config['options'],
                                index=config['options'].index(config['default']),
                                help=config['help'],
                                key=f"chain_param_{param_name}"
                            )
                            strategy_parameters[param_name] = value
            
            # Add strategy button
            if st.button("Add to Chain"):
                chain_item = {
                    'key': new_strategy_key,
                    'name': available_strategies[new_strategy_key],
                    'operator': operator,
                    'parameters': strategy_parameters
                }
                st.session_state.strategy_chain.append(chain_item)
                st.rerun()
        
        # Display current chain
        if st.session_state.strategy_chain:
            st.sidebar.subheader("📋 Current Chain")
            
            for i, item in enumerate(st.session_state.strategy_chain):
                with st.sidebar.container():
                    if i > 0:
                        st.write(f"**{item['operator']}**")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        param_str = ""
                        if item['parameters']:
                            param_str = f" ({', '.join([f'{k}={v}' for k, v in item['parameters'].items()])})"
                        st.write(f"**{i+1}.** {item['name']}{param_str}")
                    
                    with col2:
                        if st.button("❌", key=f"remove_{i}", help="Remove strategy"):
                            st.session_state.strategy_chain.pop(i)
                            st.rerun()
            
            # Clear chain button
            if st.sidebar.button("🗑️ Clear Chain"):
                st.session_state.strategy_chain = []
                st.rerun()
            
            # Set variables for screening
            selected_strategy_key = "custom_chain"
            strategy_parameters = {}
        else:
            st.sidebar.info("Add strategies to build your chain")
            selected_strategy_key = None
            strategy_parameters = {}
    
    else:
        # Single strategy mode
        # Get strategies organized by category
        try:
            strategies_by_category = strategy_manager.get_strategies_by_category()
        except Exception as e:
            st.error(f"Error loading strategies by category: {e}")
            # Fallback to basic strategy list
            strategies = strategy_manager.get_available_strategies()
            strategies_by_category = {"All": [(key, None) for key in strategies.keys()]}
        
        # Category selection
        selected_category = st.sidebar.selectbox(
            "Strategy Category:",
            list(strategies_by_category.keys()),
            help="Choose strategy category"
        )
        
        # Strategy selection within category
        category_strategies = strategies_by_category[selected_category]
        strategy_options = {key: strategy.name for key, strategy in category_strategies}
        
        selected_strategy_key = st.sidebar.selectbox(
            "Select Strategy:",
            list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            help="Choose the screening strategy to apply"
        )
        
        # Get selected strategy info
        selected_strategy = strategy_manager.get_strategy(selected_strategy_key)
        if selected_strategy:
            strategy_info = selected_strategy.get_info()
            
            # Display strategy info
            with st.sidebar.expander("ℹ️ Strategy Details", expanded=True):
                st.write(f"**Type:** {strategy_info['type'].title()}")
                st.write(f"**Description:** {strategy_info['description']}")
                if strategy_info['tags']:
                    st.write(f"**Tags:** {', '.join(strategy_info['tags'])}")
                
                # Show component strategies for composite strategies
                if hasattr(selected_strategy, 'get_strategy_details'):
                    st.write("**Component Strategies:**")
                    details = selected_strategy.get_strategy_details()
                    for detail in details:
                        st.write(f"• {detail['name']}")
            
            # Dynamic parameter configuration
            param_config = selected_strategy.get_parameter_config()
            strategy_parameters = {}
            
            if param_config:
                st.sidebar.subheader("⚙️ Strategy Parameters")
                
                for param_name, config in param_config.items():
                    if config['type'] == 'slider':
                        value = st.sidebar.slider(
                            config['label'],
                            min_value=config['min'],
                            max_value=config['max'],
                            value=config['default'],
                            step=config['step'],
                            help=config['help']
                        )
                        strategy_parameters[param_name] = value
                    
                    elif config['type'] == 'selectbox':
                        value = st.sidebar.selectbox(
                            config['label'],
                            config['options'],
                            index=config['options'].index(config['default']),
                            help=config['help']
                        )
                        strategy_parameters[param_name] = value
                    
                    elif config['type'] == 'number_input':
                        value = st.sidebar.number_input(
                            config['label'],
                            min_value=config.get('min', 0),
                            max_value=config.get('max', 1000),
                            value=config['default'],
                            step=config.get('step', 1),
                            help=config['help']
                        )
                        strategy_parameters[param_name] = value
    
    # Screen button
    screen_enabled = selected_strategy_key is not None
    if strategy_mode == "Chain Strategies":
        screen_enabled = len(st.session_state.get('strategy_chain', [])) >= 2
    
    if st.sidebar.button("🔍 Screen Stocks", type="primary", disabled=not screen_enabled):
        # Show cache status
        cache_info = screener.get_cache_info()
        if cache_info['enabled'] and cache_info['memory_entries'] > 0:
            st.info("💾 Using cached data for faster results!")
        
        with st.spinner("Screening stocks... This may take a few minutes."):
            if strategy_mode == "Chain Strategies" and st.session_state.strategy_chain:
                # Create composite strategy from chain
                chain_name = f"Custom Chain ({len(st.session_state.strategy_chain)} strategies)"
                try:
                    composite_strategy = strategy_manager.create_composite_strategy(
                        chain_name, 
                        st.session_state.strategy_chain
                    )
                    # Register temporarily
                    strategy_manager.register_strategy("custom_chain", composite_strategy)
                    results = screener.screen_stocks(selected_universe, "custom_chain")
                except Exception as e:
                    st.error(f"Error creating composite strategy: {e}")
                    results = {'success': False, 'error': str(e)}
            else:
                # Single strategy mode
                if strategy_parameters:
                    custom_strategy = strategy_manager.create_strategy_instance(selected_strategy_key, **strategy_parameters)
                    if custom_strategy:
                        # Temporarily register the custom strategy
                        temp_key = f"{selected_strategy_key}_custom"
                        strategy_manager.register_strategy(temp_key, custom_strategy)
                        results = screener.screen_stocks(selected_universe, temp_key)
                    else:
                        results = screener.screen_stocks(selected_universe, selected_strategy_key)
                else:
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
                if 'ranked_stocks' in results:
                    st.metric("Top Ranked Stocks", len(results['ranked_stocks']))
                else:
                    st.metric("Stocks Passing Strategy", len(results.get('passing_stocks', [])))
            
            st.subheader("📊 Results")
            
            # Handle momentum ranking results
            if 'ranked_stocks' in results:
                ranked_stocks = results['ranked_stocks']
                if ranked_stocks:
                    st.write("**Top 10 stocks ranked by momentum score:**")
                    
                    # Create a clean table for momentum results
                    import pandas as pd
                    
                    table_data = []
                    for i, stock in enumerate(ranked_stocks, 1):
                        table_data.append({
                            'Rank': i,
                            'Stock Code': stock['symbol'],
                            'Close Price': f"₹{stock['current_price']:.2f}",
                            'Momentum Score': f"{stock['momentum_score']:.3f}"
                        })
                    
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Option to show detailed technical indicators
                    if st.checkbox("Show detailed technical indicators"):
                        st.subheader("📊 Technical Indicators Breakdown")
                        
                        detailed_table_data = []
                        for i, stock in enumerate(ranked_stocks, 1):
                            individual = stock['individual_scores']
                            detailed_table_data.append({
                                'Rank': i,
                                'Stock Code': stock['symbol'],
                                'Close Price': f"₹{stock['current_price']:.2f}",
                                'Momentum Score': f"{stock['momentum_score']:.3f}",
                                'RSI': f"{individual.get('rsi', 0):.2f}",
                                'MACD': f"{individual.get('macd', 0):.2f}",
                                'ROC': f"{individual.get('roc', 0):.2f}",
                                'Stochastic': f"{individual.get('stochastic', 0):.2f}",
                                'Williams %R': f"{individual.get('williams_r', 0):.2f}",
                                'Moving Average': f"{individual.get('moving_average', 0):.2f}"
                            })
                        
                        detailed_df = pd.DataFrame(detailed_table_data)
                        st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Show detailed breakdown for top 3
                    st.subheader("🔍 Top 3 Stocks Analysis")
                    
                    for i, stock in enumerate(ranked_stocks[:3], 1):
                        with st.expander(f"{i}. {stock['symbol']} - Score: {stock['momentum_score']:.3f}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Technical Indicators:**")
                                individual = stock['individual_scores']
                                for indicator, score in individual.items():
                                    indicator_name = indicator.replace('_', ' ').title()
                                    st.write(f"• {indicator_name}: {score:.3f}")
                            
                            with col2:
                                st.write("**Stock Info:**")
                                st.write(f"• Current Price: ₹{stock['current_price']:.1f}")
                                st.write(f"• Overall Momentum: {stock['momentum_score']:.1%}")
                    
                    # Download option for momentum results
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Momentum Rankings (CSV)",
                        data=csv_data,
                        file_name=f"momentum_rankings_{selected_universe}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No stocks could be ranked for momentum.")
            
            # Handle regular strategy results
            elif 'passing_stocks' in results:
                passing_stocks = results['passing_stocks']
                stocks_data = results.get('stocks_data', [])
                
                if passing_stocks:
                    st.write("**Stocks that passed the screening criteria:**")
                    
                    # Create table with stock code and close price
                    import pandas as pd
                    
                    table_data = []
                    for stock_symbol in passing_stocks:
                        # Find the stock data for this symbol
                        stock_price = "N/A"
                        for stock_data in stocks_data:
                            if stock_data['symbol'] == stock_symbol:
                                quote = stock_data.get('quote', {})
                                if quote:
                                    stock_price = f"₹{quote.get('lastPrice', 0):.2f}"
                                break
                        
                        table_data.append({
                            'Stock Code': stock_symbol,
                            'Close Price': stock_price
                        })
                    
                    # Display as table
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Download option
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"screened_stocks_{selected_universe}_{selected_strategy_key}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No stocks passed the screening criteria.")
                
        else:
            st.error(f"❌ Error: {results['error']}")
    
    # Log viewer section
    if st.session_state.get('show_logs', False):
        st.markdown("---")
        st.subheader("📋 Audit Log Viewer")
        
        try:
            log_stats = screener.get_audit_log_stats()
            
            if log_stats['files']:
                # Select log file to view
                log_files = [f['name'] for f in log_stats['files']]
                selected_log = st.selectbox("Select log file:", log_files)
                
                if selected_log:
                    log_path = os.path.join(log_stats['log_dir'], selected_log)
                    
                    # Read and display log content
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        # Show last N lines
                        max_lines = st.slider("Number of lines to show:", 10, 1000, 100)
                        lines = log_content.split('\n')
                        displayed_lines = lines[-max_lines:] if len(lines) > max_lines else lines
                        
                        st.text_area(
                            f"Log content (last {len(displayed_lines)} lines):",
                            '\n'.join(displayed_lines),
                            height=400
                        )
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Full Log",
                            data=log_content,
                            file_name=selected_log,
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error reading log file: {e}")
            else:
                st.info("No log files available yet. Run a screening to generate logs.")
            
            if st.button("❌ Close Log Viewer"):
                st.session_state.show_logs = False
                st.rerun()
                
        except Exception as e:
            st.error(f"Error accessing log files: {e}")
    
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
        - **Momentum Ranking**: Ranks stocks by comprehensive momentum score using RSI, MACD, ROC, Stochastic, Williams %R, and Moving Averages
        - **Strategy Chaining**: Combine multiple strategies with AND/OR logic (e.g., "RSI > 60 AND High Volume")
        """)

if __name__ == "__main__":
    main()