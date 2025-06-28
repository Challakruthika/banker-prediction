import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from utils.bank_statement_parser import BankStatementParser
import numpy as np

# Add parent directory to path to import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.financial_models import CustomerFinancialAnalyzer
from data.sample_customers import get_all_sample_customers

class FinancialDashboard:
    """
    Interactive dashboard for financial analysis and customer insights
    Designed for bankers to make informed decisions
    """
    
    def __init__(self):
        self.analyzer = CustomerFinancialAnalyzer()
        self.customer_data = None
        self.analysis_results = []
        
    def load_csv_data(self, uploaded_file):
        """Load and validate CSV customer data"""
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = [
                'customer_id', 'customer_name', 'monthly_income', 
                'monthly_expenses', 'savings_balance', 'investment_balance',
                'total_debt', 'payment_history_score', 'credit_utilization_ratio',
                'credit_age_months'
            ]
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.info("Required columns: customer_id, customer_name, monthly_income, monthly_expenses, savings_balance, investment_balance, total_debt, payment_history_score, credit_utilization_ratio, credit_age_months")
                return None
            
            # Convert to list of dictionaries
            customers = df.to_dict('records')
            return customers
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    
    def analyze_customers(self, customers):
        """Analyze all customers and store results"""
        self.analysis_results = []
        progress_bar = st.progress(0)
        
        for i, customer in enumerate(customers):
            summary = self.analyzer.create_customer_summary(customer)
            self.analysis_results.append(summary)
            progress_bar.progress((i + 1) / len(customers))
        
        progress_bar.empty()
        return self.analysis_results
    
    def display_customer_overview(self):
        """Display high-level customer overview"""
        if not self.analysis_results:
            return
        
        st.header("📊 Customer Portfolio Overview")
        
        # Create metrics
        total_customers = len(self.analysis_results)
        avg_credit_score = sum(r['credit_score'] for r in self.analysis_results) / total_customers
        avg_risk_score = sum(r['risk_assessment']['risk_score'] for r in self.analysis_results) / total_customers
        
        # Count risk levels
        risk_counts = {}
        for result in self.analysis_results:
            risk_level = result['risk_assessment']['risk_level']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", total_customers)
        
        with col2:
            st.metric("Average Credit Score", f"{avg_credit_score:.0f}")
        
        with col3:
            st.metric("Average Risk Score", f"{avg_risk_score:.1f}")
        
        with col4:
            approved_count = sum(1 for r in self.analysis_results if r['lending_recommendations']['loan_approval'])
            st.metric("Loan Approvals", f"{approved_count}/{total_customers}")
        
        # Risk distribution chart
        st.subheader("Risk Level Distribution")
        risk_df = pd.DataFrame(list(risk_counts.items()), columns=['Risk Level', 'Count'])
        fig = px.pie(risk_df, values='Count', names='Risk Level', 
                    title="Customer Risk Distribution",
                    color_discrete_map={
                        'Low': '#00FF00',
                        'Medium': '#FFFF00', 
                        'High': '#FFA500',
                        'Very High': '#FF0000'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        if avg_credit_score < 600:
            st.warning("Average credit score is low. Consider focusing on credit-building products or stricter lending criteria.")
        else:
            st.success("Average credit score is healthy. Portfolio is well-positioned for lending.")
        if avg_risk_score > 4:
            st.error("Portfolio risk is high. Review risk management strategies.")
        if approved_count / total_customers < 0.3:
            st.info("Loan approval rate is low. Consider reviewing approval criteria or expanding target segments.")
        elif approved_count / total_customers > 0.7:
            st.info("High loan approval rate. Ensure risk controls are adequate.")
    
    def display_credit_score_analysis(self):
        """Display credit score analysis"""
        if not self.analysis_results:
            return
        
        st.header("💳 Credit Score Analysis")
        
        # Create credit score distribution
        credit_scores = [r['credit_score'] for r in self.analysis_results]
        ratings = [r['credit_rating'] for r in self.analysis_results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit score histogram
            fig = px.histogram(x=credit_scores, nbins=20, 
                             title="Credit Score Distribution",
                             labels={'x': 'Credit Score', 'y': 'Number of Customers'})
            fig.add_vline(x=580, line_dash="dash", line_color="red", 
                         annotation_text="Poor Credit Threshold")
            fig.add_vline(x=670, line_dash="dash", line_color="orange", 
                         annotation_text="Fair Credit Threshold")
            fig.add_vline(x=740, line_dash="dash", line_color="yellow", 
                         annotation_text="Good Credit Threshold")
            fig.add_vline(x=800, line_dash="dash", line_color="green", 
                         annotation_text="Exceptional Credit Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Credit rating distribution
            rating_counts = pd.Series(ratings).value_counts()
            fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                        title="Credit Rating Distribution",
                        labels={'x': 'Credit Rating', 'y': 'Number of Customers'})
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        poor = sum(1 for s in credit_scores if s < 580)
        fair = sum(1 for s in credit_scores if 580 <= s < 670)
        good = sum(1 for s in credit_scores if 670 <= s < 740)
        very_good = sum(1 for s in credit_scores if 740 <= s < 800)
        exceptional = sum(1 for s in credit_scores if s >= 800)
        if poor / len(credit_scores) > 0.5:
            st.error("Majority of customers have poor credit. Tighten lending or offer credit improvement programs.")
        elif good + very_good + exceptional > len(credit_scores) * 0.5:
            st.success("Most customers have good or better credit. Consider pre-approved offers.")
        else:
            st.info("Credit score distribution is mixed. Segment offers accordingly.")
    
    def display_financial_health_analysis(self):
        """Display financial health indicators"""
        if not self.analysis_results:
            return
        
        st.header("🏥 Financial Health Analysis")
        
        # Extract financial health data
        health_scores = [r['financial_health']['financial_health_score'] for r in self.analysis_results]
        health_categories = [r['financial_health']['health_category'] for r in self.analysis_results]
        dti_ratios = [r['financial_health']['debt_to_income_ratio'] for r in self.analysis_results]
        savings_rates = [r['financial_health']['savings_rate'] for r in self.analysis_results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Financial health score distribution
            fig = px.histogram(x=health_scores, nbins=15,
                             title="Financial Health Score Distribution",
                             labels={'x': 'Health Score', 'y': 'Number of Customers'})
            fig.add_vline(x=20, line_dash="dash", line_color="red", 
                         annotation_text="Critical")
            fig.add_vline(x=40, line_dash="dash", line_color="orange", 
                         annotation_text="Poor")
            fig.add_vline(x=60, line_dash="dash", line_color="yellow", 
                         annotation_text="Fair")
            fig.add_vline(x=80, line_dash="dash", line_color="green", 
                         annotation_text="Excellent")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # DTI vs Savings Rate scatter plot
            fig = px.scatter(x=dti_ratios, y=savings_rates,
                           title="Debt-to-Income vs Savings Rate",
                           labels={'x': 'Debt-to-Income Ratio', 'y': 'Savings Rate'},
                           color=health_categories)
            fig.add_hline(y=0.1, line_dash="dash", line_color="green", 
                         annotation_text="Good Savings Rate (10%)")
            fig.add_vline(x=0.28, line_dash="dash", line_color="green", 
                         annotation_text="Good DTI (28%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        low_health = sum(1 for s in health_scores if s < 40)
        high_health = sum(1 for s in health_scores if s >= 60)
        if low_health / len(health_scores) > 0.5:
            st.error("Most customers have poor financial health. Consider offering financial wellness programs.")
        elif high_health / len(health_scores) > 0.5:
            st.success("Majority have good financial health. Portfolio is resilient.")
        else:
            st.info("Financial health is mixed. Target interventions where needed.")
    
    def display_lending_recommendations(self):
        """Display lending recommendations summary"""
        if not self.analysis_results:
            return
        
        st.header("💰 Lending Recommendations")
        
        # Filter approved customers
        approved_customers = [r for r in self.analysis_results if r['lending_recommendations']['loan_approval']]
        rejected_customers = [r for r in self.analysis_results if not r['lending_recommendations']['loan_approval']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Approved Customers")
            if approved_customers:
                approved_df = pd.DataFrame([
                    {
                        'Customer': r['customer_name'],
                        'Credit Score': r['credit_score'],
                        'Risk Level': r['risk_assessment']['risk_level'],
                        'Recommended Amount': f"${r['lending_recommendations']['recommended_loan_amount']:,.0f}",
                        'Interest Rate': r['lending_recommendations']['interest_rate_range']
                    }
                    for r in approved_customers
                ])
                st.dataframe(approved_df, use_container_width=True)
            else:
                st.info("No customers approved for loans")
        
        with col2:
            st.subheader("❌ Rejected Customers")
            if rejected_customers:
                rejection_reasons = []
                for r in rejected_customers:
                    reasons = []
                    if r['credit_score'] < 700:
                        reasons.append("Low credit score")
                    if r['risk_assessment']['risk_level'] in ['High', 'Very High']:
                        reasons.append("High risk level")
                    rejection_reasons.append(", ".join(reasons))
                
                rejected_df = pd.DataFrame([
                    {
                        'Customer': r['customer_name'],
                        'Credit Score': r['credit_score'],
                        'Risk Level': r['risk_assessment']['risk_level'],
                        'Rejection Reasons': reason
                    }
                    for r, reason in zip(rejected_customers, rejection_reasons)
                ])
                st.dataframe(rejected_df, use_container_width=True)
            else:
                st.info("All customers approved for loans")
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        approved_customers = [r for r in self.analysis_results if r['lending_recommendations']['loan_approval']]
        rejected_customers = [r for r in self.analysis_results if not r['lending_recommendations']['loan_approval']]
        if not approved_customers:
            st.warning("No customers approved for loans. Review approval criteria or customer targeting.")
        if rejected_customers:
            common_reasons = []
            for r in rejected_customers:
                if r['credit_score'] < 700:
                    common_reasons.append("Low credit score")
                if r['risk_assessment']['risk_level'] in ['High', 'Very High']:
                    common_reasons.append("High risk level")
            if common_reasons:
                st.info(f"Common rejection reasons: {', '.join(set(common_reasons))}")
    
    def display_individual_customer_analysis(self):
        """Display detailed analysis for individual customers"""
        if not self.analysis_results:
            return
        
        st.header("👤 Individual Customer Analysis")
        
        # Customer selector
        customer_options = [f"{r['customer_name']} ({r['customer_id']})" for r in self.analysis_results]
        selected_customer = st.selectbox("Select Customer", customer_options)
        
        if selected_customer:
            # Find selected customer
            customer_id = selected_customer.split("(")[-1].split(")")[0]
            customer_result = next(r for r in self.analysis_results if r['customer_id'] == customer_id)
            
            # Display customer details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Information")
                st.write(f"**Name:** {customer_result['customer_name']}")
                st.write(f"**Customer ID:** {customer_result['customer_id']}")
                st.write(f"**Credit Score:** {customer_result['credit_score']} ({customer_result['credit_rating']})")
                st.write(f"**Risk Level:** {customer_result['risk_assessment']['risk_level']}")
                st.write(f"**Financial Health:** {customer_result['financial_health']['health_category']}")
                
                # Key metrics
                st.subheader("Key Financial Metrics")
                metrics = customer_result['key_metrics']
                st.write(f"**Monthly Income:** ${metrics['monthly_income']:,.0f}")
                st.write(f"**Monthly Expenses:** ${metrics['monthly_expenses']:,.0f}")
                st.write(f"**Total Debt:** ${metrics['total_debt']:,.0f}")
                st.write(f"**Savings Balance:** ${metrics['savings_balance']:,.0f}")
                st.write(f"**Investment Balance:** ${metrics['investment_balance']:,.0f}")
            
            with col2:
                # Financial health indicators
                st.subheader("Financial Health Indicators")
                health = customer_result['financial_health']
                st.write(f"**Emergency Fund Ratio:** {health['emergency_fund_ratio']:.1f} months")
                st.write(f"**Savings Rate:** {health['savings_rate']:.1%}")
                st.write(f"**Debt-to-Income Ratio:** {health['debt_to_income_ratio']:.1%}")
                st.write(f"**Investment Ratio:** {health['investment_ratio']:.1%}")
                st.write(f"**Net Worth:** ${health['net_worth']:,.0f}")
                
                # Lending recommendations
                st.subheader("Lending Decision")
                lending = customer_result['lending_recommendations']
                if lending['loan_approval']:
                    st.success("✅ **APPROVED**")
                    st.write(f"**Recommended Amount:** ${lending['recommended_loan_amount']:,.0f}")
                    st.write(f"**Interest Rate Range:** {lending['interest_rate_range']}")
                else:
                    st.error("❌ **REJECTED**")
                
                if lending['conditions']:
                    st.write("**Conditions:**")
                    for condition in lending['conditions']:
                        st.write(f"• {condition}")
                
                if lending['risk_mitigation']:
                    st.write("**Risk Mitigation:**")
                    for mitigation in lending['risk_mitigation']:
                        st.write(f"• {mitigation}")
            
            # --- Insights & Suggestions ---
            st.markdown("---")
            st.subheader("Insights & Suggestions")
            if customer_result['credit_score'] >= 740 and customer_result['risk_assessment']['risk_level'] == 'Low':
                st.success("Excellent candidate for premium products or pre-approved loans.")
            elif customer_result['risk_assessment']['risk_level'] in ['High', 'Very High']:
                st.error("High risk customer. Recommend additional due diligence or risk mitigation.")
            elif customer_result['financial_health']['health_category'] in ['Poor', 'Critical']:
                st.warning("Customer has poor financial health. Suggest financial counseling or restricted lending.")
            else:
                st.info("Customer is in a moderate segment. Review full profile before proceeding.")
    
    def display_insights_and_segmentation(self):
        """Display advanced insights and customer segmentation"""
        if not self.analysis_results:
            return
        st.header("🔍 Insights & Customer Segmentation")
        customer = self.analysis_results[0]  # Only one customer for bank statement mode
        metrics = customer['key_metrics']
        health = customer['financial_health']
        
        # Savings rate
        savings_rate = health['savings_rate']
        net_cash_flow = metrics['monthly_income'] - metrics['monthly_expenses']
        
        # Segmentation logic
        income = metrics['monthly_income']
        expenses = metrics['monthly_expenses']
        balance = metrics['savings_balance']
        
        if income >= 50000:
            income_segment = "High Income"
        elif income >= 20000:
            income_segment = "Moderate Income"
        else:
            income_segment = "Low Income"
        
        if savings_rate >= 0.2:
            savings_segment = "Saver"
        elif savings_rate >= 0.05:
            savings_segment = "Balanced"
        else:
            savings_segment = "High Spender"
        
        if abs(net_cash_flow) < 0.05 * income:
            cashflow_segment = "Stable Cash Flow"
        elif net_cash_flow > 0:
            cashflow_segment = "Net Saver"
        else:
            cashflow_segment = "Net Spender"
        
        st.subheader("Customer Segmentation")
        st.write(f"**Income Segment:** {income_segment}")
        st.write(f"**Spending/Saving Segment:** {savings_segment}")
        st.write(f"**Cash Flow Segment:** {cashflow_segment}")
        
        st.markdown("---")
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Income", f"₹{income:,.0f}")
        with col2:
            st.metric("Monthly Expenses", f"₹{expenses:,.0f}")
        with col3:
            st.metric("Net Cash Flow", f"₹{net_cash_flow:,.0f}", delta=f"{savings_rate:.1%} Savings Rate")
        st.write(f"**Savings Rate:** {savings_rate:.1%}")
        st.write(f"**Current Balance:** ₹{balance:,.0f}")
        
        # Trend chart if possible
        if 'analysis_date' in customer:
            st.info("Trends are based on the uploaded statement period. For multi-month trends, upload a longer statement.")
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations for Banker")
        if income_segment == "High Income" and savings_segment == "Saver":
            st.success("This customer is financially strong and a low-risk candidate for most banking products.")
        elif savings_segment == "High Spender":
            st.warning("Customer spends most of their income. Consider counseling on savings or offering budgeting tools.")
        elif cashflow_segment == "Net Spender":
            st.error("Customer is spending more than they earn. High risk for lending.")
        else:
            st.info("Customer is in a moderate segment. Review full profile before proceeding.")
    
    def export_results(self):
        """Export analysis results to Excel"""
        if not self.analysis_results:
            return
        
        # Create comprehensive Excel report
        with pd.ExcelWriter('financial_analysis_report.xlsx', engine='xlsxwriter') as writer:
            
            # Summary sheet
            summary_data = []
            for r in self.analysis_results:
                summary_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Credit Score': r['credit_score'],
                    'Credit Rating': r['credit_rating'],
                    'Risk Level': r['risk_assessment']['risk_level'],
                    'Risk Score': r['risk_assessment']['risk_score'],
                    'Financial Health Score': r['financial_health']['financial_health_score'],
                    'Health Category': r['financial_health']['health_category'],
                    'Loan Approval': r['lending_recommendations']['loan_approval'],
                    'Recommended Amount': r['lending_recommendations']['recommended_loan_amount'],
                    'Interest Rate Range': r['lending_recommendations']['interest_rate_range'],
                    'Monthly Income': r['key_metrics']['monthly_income'],
                    'Monthly Expenses': r['key_metrics']['monthly_expenses'],
                    'Total Debt': r['key_metrics']['total_debt'],
                    'Savings Balance': r['key_metrics']['savings_balance'],
                    'Investment Balance': r['key_metrics']['investment_balance'],
                    'DTI Ratio': r['financial_health']['debt_to_income_ratio'],
                    'Savings Rate': r['financial_health']['savings_rate'],
                    'Emergency Fund Ratio': r['financial_health']['emergency_fund_ratio'],
                    'Net Worth': r['financial_health']['net_worth']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk factors sheet
            risk_data = []
            for r in self.analysis_results:
                risk_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Risk Level': r['risk_assessment']['risk_level'],
                    'Risk Factors': ', '.join(r['risk_assessment']['risk_factors'])
                })
            
            risk_df = pd.DataFrame(risk_data)
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # Lending recommendations sheet
            lending_data = []
            for r in self.analysis_results:
                lending_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Loan Approval': r['lending_recommendations']['loan_approval'],
                    'Recommended Amount': r['lending_recommendations']['recommended_loan_amount'],
                    'Interest Rate Range': r['lending_recommendations']['interest_rate_range'],
                    'Loan Terms': ', '.join(r['lending_recommendations']['loan_terms']),
                    'Conditions': ', '.join(r['lending_recommendations']['conditions']),
                    'Risk Mitigation': ', '.join(r['lending_recommendations']['risk_mitigation'])
                })
            
            lending_df = pd.DataFrame(lending_data)
            lending_df.to_excel(writer, sheet_name='Lending Recommendations', index=False)
        
        return 'financial_analysis_report.xlsx'

    def load_bank_statement(self, uploaded_file, customer_id='AUTO', customer_name='Unknown'):
        """Load and parse a raw bank statement CSV file"""
        try:
            parser = BankStatementParser(customer_id=customer_id, customer_name=customer_name)
            # Save uploaded file to a temp location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            customer = parser.parse(tmp_path)
            return [customer]
        except Exception as e:
            st.error(f"Error parsing bank statement: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Banker's Financial Insights Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Banker's Financial Insights Dashboard")
    st.markdown("### Comprehensive Customer Analysis for Informed Banking Decisions")
    
    dashboard = FinancialDashboard()
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Input")
    
    upload_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Upload Bank Statement", "Use Sample Data"]
    )
    
    customers = None
    if upload_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload customer data CSV file",
            type=['csv'],
            help="CSV should contain: customer_id, customer_name, monthly_income, monthly_expenses, savings_balance, investment_balance, total_debt, payment_history_score, credit_utilization_ratio, credit_age_months"
        )
        if uploaded_file is not None:
            customers = dashboard.load_csv_data(uploaded_file)
            if customers:
                st.sidebar.success(f"✅ Loaded {len(customers)} customers")
                dashboard.analyze_customers(customers)
            else:
                st.sidebar.error("❌ Failed to load CSV data")
                return
    elif upload_option == "Upload Bank Statement":
        uploaded_file = st.sidebar.file_uploader(
            "Upload raw bank statement CSV file",
            type=['csv'],
            help="Supported: PNB, SBI, ICICI, APGB statement CSVs"
        )
        customer_id = st.sidebar.text_input("Customer ID (optional)", value="AUTO")
        customer_name = st.sidebar.text_input("Customer Name (optional)", value="Unknown")
        if uploaded_file is not None:
            customers = dashboard.load_bank_statement(uploaded_file, customer_id, customer_name)
            if customers:
                st.sidebar.success(f"✅ Parsed bank statement for {customers[0]['customer_name']}")
                dashboard.analyze_customers(customers)
            else:
                st.sidebar.error("❌ Failed to parse bank statement")
                return
    else:
        customers = get_all_sample_customers()
        st.sidebar.success(f"✅ Loaded {len(customers)} sample customers")
        dashboard.analyze_customers(customers)
    
    if dashboard.analysis_results:
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "💳 Credit Analysis", 
            "🏥 Financial Health", 
            "💰 Lending Decisions",
            "👤 Individual Analysis",
            "🔍 Insights & Segmentation"
        ])
        
        with tab1:
            dashboard.display_customer_overview()
        
        with tab2:
            dashboard.display_credit_score_analysis()
        
        with tab3:
            dashboard.display_financial_health_analysis()
        
        with tab4:
            dashboard.display_lending_recommendations()
        
        with tab5:
            dashboard.display_individual_customer_analysis()
        
        with tab6:
            dashboard.display_insights_and_segmentation()
        
        # Export functionality
        st.sidebar.header("📤 Export Results")
        if st.sidebar.button("Export to Excel"):
            filename = dashboard.export_results()
            with open(filename, 'rb') as f:
                st.sidebar.download_button(
                    label="Download Excel Report",
                    data=f.read(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main() 