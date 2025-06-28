#!/usr/bin/env python3
"""
Banker's Financial Insights Model
Main application entry point

This application provides comprehensive financial analysis for banking customers,
helping bankers make informed decisions about lending and customer relationships.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.financial_models import CustomerFinancialAnalyzer
from utils.csv_processor import CSVProcessor
from data.sample_customers import get_all_sample_customers

def run_analysis_from_csv(csv_file_path: str, output_file: str = None):
    """
    Run financial analysis on customer data from CSV file
    """
    print("🏦 Banker's Financial Insights Model")
    print("=" * 50)
    
    # Initialize components
    processor = CSVProcessor()
    analyzer = CustomerFinancialAnalyzer()
    
    # Validate CSV file
    print(f"📁 Validating CSV file: {csv_file_path}")
    validation_result = processor.validate_csv(csv_file_path)
    
    if not validation_result['is_valid']:
        print("❌ CSV validation failed:")
        for error in validation_result['errors']:
            print(f"   - {error}")
        return False
    
    print("✅ CSV validation passed")
    print(f"   Data quality score: {validation_result['data_quality_score']:.1%}")
    
    if validation_result['warnings']:
        print("⚠️  Warnings:")
        for warning in validation_result['warnings']:
            print(f"   - {warning}")
    
    # Process CSV data
    print("\n📊 Processing customer data...")
    try:
        customers = processor.process_csv(csv_file_path)
        print(f"✅ Processed {len(customers)} customers")
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return False
    
    # Run analysis
    print("\n🔍 Running financial analysis...")
    analysis_results = []
    
    for i, customer in enumerate(customers, 1):
        print(f"   Analyzing customer {i}/{len(customers)}: {customer.get('customer_name', 'Unknown')}")
        summary = analyzer.create_customer_summary(customer)
        analysis_results.append(summary)
    
    # Display summary
    print("\n📈 Analysis Summary:")
    print("-" * 30)
    
    total_customers = len(analysis_results)
    approved_count = sum(1 for r in analysis_results if r['lending_recommendations']['loan_approval'])
    avg_credit_score = sum(r['credit_score'] for r in analysis_results) / total_customers
    
    print(f"Total Customers: {total_customers}")
    print(f"Loan Approvals: {approved_count}/{total_customers} ({approved_count/total_customers:.1%})")
    print(f"Average Credit Score: {avg_credit_score:.0f}")
    
    # Risk distribution
    risk_counts = {}
    for result in analysis_results:
        risk_level = result['risk_assessment']['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
    
    print("\nRisk Level Distribution:")
    for risk_level, count in risk_counts.items():
        print(f"  {risk_level}: {count} customers ({count/total_customers:.1%})")
    
    # Save results
    if output_file:
        print(f"\n💾 Saving results to: {output_file}")
        save_analysis_results(analysis_results, output_file)
    
    return True

def save_analysis_results(analysis_results, output_file: str):
    """
    Save analysis results to Excel file
    """
    import pandas as pd
    
    # Create comprehensive Excel report
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        
        # Summary sheet
        summary_data = []
        for r in analysis_results:
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
        
        # Risk analysis sheet
        risk_data = []
        for r in analysis_results:
            risk_data.append({
                'Customer ID': r['customer_id'],
                'Customer Name': r['customer_name'],
                'Risk Level': r['risk_assessment']['risk_level'],
                'Risk Score': r['risk_assessment']['risk_score'],
                'Risk Factors': ', '.join(r['risk_assessment']['risk_factors'])
            })
        
        risk_df = pd.DataFrame(risk_data)
        risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
        
        # Lending recommendations sheet
        lending_data = []
        for r in analysis_results:
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
    
    print(f"✅ Results saved successfully to {output_file}")

def create_sample_csv(output_path: str = "sample_customers.csv", num_customers: int = 10):
    """
    Create a sample CSV file for testing
    """
    processor = CSVProcessor()
    processor.create_sample_csv(output_path, num_customers)
    print(f"✅ Sample CSV file created: {output_path}")
    print(f"   Contains {num_customers} sample customers")

def show_csv_template():
    """
    Display CSV template format
    """
    processor = CSVProcessor()
    template = processor.get_csv_template()
    
    print("📋 CSV Template Format:")
    print("=" * 50)
    print(template)
    
    print("\n📝 Required Columns:")
    for col in processor.required_columns:
        print(f"   - {col}: {processor.column_descriptions[col]}")
    
    print("\n📝 Optional Columns:")
    for col in processor.optional_columns:
        print(f"   - {col}: {processor.column_descriptions[col]}")

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(
        description="Banker's Financial Insights Model - Analyze customer financial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --csv customers.csv --output results.xlsx
  python main.py --create-sample sample_data.csv
  python main.py --template
  python main.py --dashboard
        """
    )
    
    parser.add_argument(
        '--csv', 
        type=str, 
        help='Path to CSV file with customer data'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='financial_analysis_results.xlsx',
        help='Output Excel file path (default: financial_analysis_results.xlsx)'
    )
    
    parser.add_argument(
        '--create-sample', 
        type=str, 
        help='Create a sample CSV file with the specified path'
    )
    
    parser.add_argument(
        '--num-customers', 
        type=int, 
        default=10,
        help='Number of customers in sample CSV (default: 10)'
    )
    
    parser.add_argument(
        '--template', 
        action='store_true',
        help='Show CSV template format'
    )
    
    parser.add_argument(
        '--dashboard', 
        action='store_true',
        help='Launch interactive dashboard'
    )
    
    args = parser.parse_args()
    
    if args.template:
        show_csv_template()
        return
    
    if args.create_sample:
        create_sample_csv(args.create_sample, args.num_customers)
        return
    
    if args.dashboard:
        print("🚀 Launching interactive dashboard...")
        print("   Please run: streamlit run dashboard/financial_dashboard.py")
        return
    
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"❌ CSV file not found: {args.csv}")
            return
        
        success = run_analysis_from_csv(args.csv, args.output)
        if success:
            print(f"\n✅ Analysis completed successfully!")
            print(f"   Results saved to: {args.output}")
        else:
            print("\n❌ Analysis failed!")
            return
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n💡 Quick Start:")
        print("   1. Create sample data: python main.py --create-sample my_customers.csv")
        print("   2. Run analysis: python main.py --csv my_customers.csv")
        print("   3. Launch dashboard: python main.py --dashboard")

if __name__ == "__main__":
    main() 