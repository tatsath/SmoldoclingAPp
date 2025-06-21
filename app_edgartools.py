import streamlit as st
import pandas as pd
import os
import boto3
import json
from pathlib import Path
import tempfile
import shutil
from edgar import set_identity, Company
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="EDGAR Tools - SEC Financial Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EDGAR Tools - SEC Financial Data Explorer")
st.markdown("Explore company financial data, filings, and insights using the SEC's EDGAR database")

# --- AWS Bedrock Setup ---

def enable_bedrock(region='us-east-1'):
    """Enable AWS Bedrock using credentials from environment variables."""
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get('AWS_REGION', region),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
    )

def bedrock_chat(prompt, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
    """Invoke Bedrock model for chat completion."""
    try:
        bedrock_runtime = enable_bedrock()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response['body'].read())['content'][0]['text']
    except Exception as e:
        st.error(f"Error communicating with Bedrock: {e}")
        return None

# --- EDGAR Identity Setup ---

st.sidebar.header("🔐 EDGAR Identity Configuration")
st.sidebar.info("The SEC requires all API users to provide a name and email address.")

user_name = st.sidebar.text_input("Your Name", "Vincent Gregoire")
user_email = st.sidebar.text_input("Your Email", "vincent@codes.finance")
user_identity = f"{user_name} {user_email}"

if st.sidebar.button("Set EDGAR Identity"):
    try:
        set_identity(user_identity)
        st.sidebar.success(f"✅ Identity set to: {user_identity}")
        st.session_state.identity_set = True
    except Exception as e:
        st.sidebar.error(f"❌ Failed to set identity: {e}")

# Check if identity is set
if not st.session_state.get('identity_set', False):
    st.warning("⚠️ Please set your EDGAR identity in the sidebar before proceeding.")
    st.stop()

# --- Main Application ---

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏢 Company Overview", 
    "📈 Financial Data", 
    "📋 Filings Explorer", 
    "🤖 AI Analysis", 
    "📊 Data Export"
])

# Tab 1: Company Overview
with tab1:
    st.header("🏢 Company Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_input = st.text_input(
            "Enter Company Ticker or CIK:",
            placeholder="e.g., AAPL, MSFT, or 320193",
            help="You can use either the company ticker symbol or CIK number"
        )
    
    with col2:
        if st.button("🔍 Load Company Data", type="primary"):
            if company_input:
                try:
                    with st.spinner("Loading company data..."):
                        company = Company(company_input)
                        st.session_state.company = company
                        st.session_state.company_loaded = True
                    st.success(f"✅ Successfully loaded {company_input.upper()}")
                except Exception as e:
                    st.error(f"❌ Error loading company: {e}")
    
    # Display company information
    if st.session_state.get('company_loaded', False) and st.session_state.get('company'):
        company = st.session_state.company
        
        # Company basic info
        st.subheader("📋 Company Information")
        
        # Create a nice display of company info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CIK", company.cik)
            st.metric("Category", company.category)
        
        with col2:
            st.metric("Industry", company.industry)
            st.metric("Incorporated", company.incorporated)
        
        with col3:
            st.metric("Business Address", company.business_address)
            st.metric("Mailing Address", company.mailing_address)
        
        # Former names
        if hasattr(company, 'former_names') and company.former_names:
            st.subheader("📝 Former Names")
            former_names_df = pd.DataFrame(company.former_names)
            st.dataframe(former_names_df, use_container_width=True)

# Tab 2: Financial Data
with tab2:
    st.header("📈 Financial Data Analysis")
    
    if not st.session_state.get('company_loaded', False):
        st.info("👆 Please load a company first in the Company Overview tab.")
    else:
        company = st.session_state.company
        
        try:
            # Get financial data
            financials = company.financials
            
            if financials:
                st.subheader("💰 Financial Statements")
                
                # Create tabs for different financial statements
                fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])
                
                with fin_tab1:
                    st.subheader("📊 Balance Sheet")
                    if hasattr(financials, 'balance_sheet') and financials.balance_sheet is not None:
                        st.dataframe(financials.balance_sheet, use_container_width=True)
                    else:
                        st.info("Balance sheet data not available")
                
                with fin_tab2:
                    st.subheader("📈 Income Statement")
                    if hasattr(financials, 'income_statement') and financials.income_statement is not None:
                        st.dataframe(financials.income_statement, use_container_width=True)
                    else:
                        st.info("Income statement data not available")
                
                with fin_tab3:
                    st.subheader("💸 Cash Flow Statement")
                    if hasattr(financials, 'cash_flow') and financials.cash_flow is not None:
                        st.dataframe(financials.cash_flow, use_container_width=True)
                    else:
                        st.info("Cash flow statement data not available")
                
                # Financial metrics summary
                st.subheader("📊 Key Financial Metrics")
                
                # Try to extract key metrics
                try:
                    if hasattr(financials, 'balance_sheet') and financials.balance_sheet is not None:
                        bs_data = financials.balance_sheet
                        if isinstance(bs_data, pd.DataFrame) and not bs_data.empty:
                            # Extract key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if 'Total Assets' in bs_data.index:
                                    total_assets = bs_data.loc['Total Assets'].iloc[0]
                                    st.metric("Total Assets", f"${total_assets:,.0f}")
                            
                            with col2:
                                if 'Total Liabilities' in bs_data.index:
                                    total_liabilities = bs_data.loc['Total Liabilities'].iloc[0]
                                    st.metric("Total Liabilities", f"${total_liabilities:,.0f}")
                            
                            with col3:
                                if 'Total Stockholders\' Equity' in bs_data.index:
                                    equity = bs_data.loc['Total Stockholders\' Equity'].iloc[0]
                                    st.metric("Stockholders' Equity", f"${equity:,.0f}")
                            
                            with col4:
                                if 'Cash and Cash Equivalents' in bs_data.index:
                                    cash = bs_data.loc['Cash and Cash Equivalents'].iloc[0]
                                    st.metric("Cash & Equivalents", f"${cash:,.0f}")
                except Exception as e:
                    st.info("Key metrics extraction not available")
                    
        except Exception as e:
            st.error(f"❌ Error loading financial data: {e}")

# Tab 3: Filings Explorer
with tab3:
    st.header("📋 Filings Explorer")
    
    if not st.session_state.get('company_loaded', False):
        st.info("👆 Please load a company first in the Company Overview tab.")
    else:
        company = st.session_state.company
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            filing_types = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "S-3", "424B2", "424B3", "424B4", "424B5"]
            selected_filing_type = st.selectbox("Select Filing Type:", filing_types)
        
        with col2:
            date_range = st.selectbox("Date Range:", ["Last 5 years", "Last 3 years", "Last year", "All available"])
        
        with col3:
            if st.button("🔍 Get Filings", type="primary"):
                try:
                    with st.spinner("Retrieving filings..."):
                        # Map date range to count
                        date_map = {
                            "Last 5 years": 60,
                            "Last 3 years": 36,
                            "Last year": 12,
                            "All available": 100
                        }
                        count = date_map.get(date_range, 20)
                        
                        filings = company.get_filings(form=selected_filing_type, count=count)
                        st.session_state.filings = filings
                        st.session_state.filings_loaded = True
                    st.success(f"✅ Retrieved {len(filings)} {selected_filing_type} filings")
                except Exception as e:
                    st.error(f"❌ Error retrieving filings: {e}")
        
        # Display filings
        if st.session_state.get('filings_loaded', False) and st.session_state.get('filings'):
            filings = st.session_state.filings
            
            st.subheader(f"📄 {selected_filing_type} Filings")
            
            # Convert to pandas for better display
            try:
                filings_df = filings.to_pandas()
                st.dataframe(filings_df, use_container_width=True)
                
                # Filing statistics
                st.subheader("📊 Filing Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Filings", len(filings_df))
                
                with col2:
                    if 'filed' in filings_df.columns:
                        latest_filing = filings_df['filed'].max()
                        st.metric("Latest Filing", latest_filing)
                
                with col3:
                    if 'filed' in filings_df.columns:
                        earliest_filing = filings_df['filed'].min()
                        st.metric("Earliest Filing", earliest_filing)
                
                with col4:
                    if 'xbrl' in filings_df.columns:
                        xbrl_count = filings_df['xbrl'].sum()
                        st.metric("XBRL Available", f"{xbrl_count}/{len(filings_df)}")
                
            except Exception as e:
                st.error(f"❌ Error processing filings data: {e}")

# Tab 4: AI Analysis
with tab4:
    st.header("🤖 AI-Powered Financial Analysis")
    
    if not st.session_state.get('company_loaded', False):
        st.info("👆 Please load a company first in the Company Overview tab.")
    else:
        company = st.session_state.company
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Financial Health Assessment", "Trend Analysis", "Risk Assessment", "Custom Query"]
        )
        
        if analysis_type == "Custom Query":
            custom_prompt = st.text_area(
                "Enter your custom analysis question:",
                placeholder="e.g., Analyze the company's debt-to-equity ratio trends over the past 5 years",
                height=100
            )
        else:
            custom_prompt = ""
        
        if st.button("🔍 Run AI Analysis", type="primary"):
            try:
                with st.spinner("Running AI analysis..."):
                    # Prepare context data
                    context_data = f"""
                    Company: {company.ticker} ({company.cik})
                    Industry: {company.industry}
                    Category: {company.category}
                    """
                    
                    # Add financial data if available
                    try:
                        financials = company.financials
                        if financials:
                            context_data += f"\nFinancial Data Available: Yes"
                    except:
                        context_data += f"\nFinancial Data Available: No"
                    
                    # Create analysis prompt
                    if analysis_type == "Financial Health Assessment":
                        prompt = f"""
                        You are a senior financial analyst. Analyze the financial health of {company.ticker} based on the following information:
                        
                        {context_data}
                        
                        Please provide:
                        1. Overall financial health score (1-10)
                        2. Key strengths and weaknesses
                        3. Risk factors to consider
                        4. Recommendations for investors
                        
                        Be specific and provide actionable insights.
                        """
                    elif analysis_type == "Trend Analysis":
                        prompt = f"""
                        You are a financial analyst specializing in trend analysis. Analyze the trends for {company.ticker}:
                        
                        {context_data}
                        
                        Please provide:
                        1. Revenue and profit trends
                        2. Balance sheet trends
                        3. Cash flow patterns
                        4. Future outlook based on trends
                        
                        Focus on identifying patterns and their implications.
                        """
                    elif analysis_type == "Risk Assessment":
                        prompt = f"""
                        You are a risk analyst. Assess the risks associated with {company.ticker}:
                        
                        {context_data}
                        
                        Please provide:
                        1. Financial risks
                        2. Operational risks
                        3. Market risks
                        4. Regulatory risks
                        5. Risk mitigation strategies
                        
                        Rate each risk category as Low, Medium, or High.
                        """
                    else:  # Custom Query
                        prompt = f"""
                        You are a financial analyst. Answer the following question about {company.ticker}:
                        
                        {context_data}
                        
                        Question: {custom_prompt}
                        
                        Provide a comprehensive and well-reasoned answer.
                        """
                    
                    # Get AI response
                    response = bedrock_chat(prompt)
                    
                    if response:
                        st.subheader("🤖 AI Analysis Results")
                        st.markdown(response)
                        
                        # Save to session state
                        if 'ai_analyses' not in st.session_state:
                            st.session_state.ai_analyses = []
                        
                        st.session_state.ai_analyses.append({
                            'company': company.ticker,
                            'type': analysis_type,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'response': response
                        })
                    else:
                        st.error("❌ Failed to generate AI analysis")
                        
            except Exception as e:
                st.error(f"❌ Error during AI analysis: {e}")
        
        # Show previous analyses
        if st.session_state.get('ai_analyses'):
            st.subheader("📚 Previous Analyses")
            for i, analysis in enumerate(reversed(st.session_state.ai_analyses)):
                with st.expander(f"{analysis['timestamp']} - {analysis['company']} ({analysis['type']})"):
                    st.markdown(analysis['response'])

# Tab 5: Data Export
with tab5:
    st.header("📊 Data Export")
    
    if not st.session_state.get('company_loaded', False):
        st.info("👆 Please load a company first in the Company Overview tab.")
    else:
        company = st.session_state.company
        
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Company Information**")
            if st.button("📄 Export Company Data (CSV)"):
                try:
                    # Create company data
                    company_data = {
                        'Field': ['Ticker', 'CIK', 'Category', 'Industry', 'Incorporated', 'Business Address', 'Mailing Address'],
                        'Value': [
                            company.ticker,
                            company.cik,
                            company.category,
                            company.industry,
                            company.incorporated,
                            company.business_address,
                            company.mailing_address
                        ]
                    }
                    company_df = pd.DataFrame(company_data)
                    
                    # Download
                    csv = company_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Company Data",
                        data=csv,
                        file_name=f"{company.ticker}_company_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error exporting company data: {e}")
        
        with col2:
            st.write("**Financial Data**")
            if st.button("📈 Export Financial Data (CSV)"):
                try:
                    financials = company.financials
                    if financials:
                        # Try to export financial data
                        if hasattr(financials, 'balance_sheet') and financials.balance_sheet is not None:
                            bs_df = financials.balance_sheet
                            if isinstance(bs_df, pd.DataFrame):
                                csv = bs_df.to_csv()
                                st.download_button(
                                    label="📥 Download Balance Sheet",
                                    data=csv,
                                    file_name=f"{company.ticker}_balance_sheet.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.info("No financial data available for export")
                except Exception as e:
                    st.error(f"❌ Error exporting financial data: {e}")
        
        # Export filings if available
        if st.session_state.get('filings_loaded', False) and st.session_state.get('filings'):
            st.subheader("📋 Export Filings Data")
            
            try:
                filings_df = st.session_state.filings.to_pandas()
                csv = filings_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filings Data",
                    data=csv,
                    file_name=f"{company.ticker}_filings_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Error exporting filings data: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📊 Powered by EDGAR Tools | SEC Financial Data Explorer</p>
        <p>Data sourced from the U.S. Securities and Exchange Commission (SEC) EDGAR database</p>
    </div>
    """,
    unsafe_allow_html=True
) 