#!/usr/bin/env python3
"""
Test script to demonstrate the Banker's Financial Insights Model
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.financial_models import CustomerFinancialAnalyzer
from data.sample_customers import get_sample_customer, get_all_sample_customers

def test_single_customer():
    """Test analysis on a single customer"""
    print("🧪 Testing Single Customer Analysis")
    print("=" * 50)
    
    # Get a sample customer
    customer = get_sample_customer('CUST001')
    
    # Initialize analyzer
    analyzer = CustomerFinancialAnalyzer()
    
    # Create comprehensive summary
    summary = analyzer.create_customer_summary(customer)
    
    # Display results
    print(f"Customer: {summary['customer_name']} ({summary['customer_id']})")
    print(f"Credit Score: {summary['credit_score']} ({summary['credit_rating']})")
    print(f"Risk Level: {summary['risk_assessment']['risk_level']}")
    print(f"Financial Health: {summary['financial_health']['health_category']}")
    print(f"Loan Approval: {'✅ APPROVED' if summary['lending_recommendations']['loan_approval'] else '❌ REJECTED'}")
    
    if summary['lending_recommendations']['loan_approval']:
        print(f"Recommended Amount: ${summary['lending_recommendations']['recommended_loan_amount']:,.0f}")
        print(f"Interest Rate Range: {summary['lending_recommendations']['interest_rate_range']}")
    
    print("\nKey Financial Metrics:")
    metrics = summary['key_metrics']
    print(f"  Monthly Income: ${metrics['monthly_income']:,.0f}")
    print(f"  Monthly Expenses: ${metrics['monthly_expenses']:,.0f}")
    print(f"  Total Debt: ${metrics['total_debt']:,.0f}")
    print(f"  Savings Balance: ${metrics['savings_balance']:,.0f}")
    print(f"  Investment Balance: ${metrics['investment_balance']:,.0f}")
    
    print("\nFinancial Health Indicators:")
    health = summary['financial_health']
    print(f"  Emergency Fund Ratio: {health['emergency_fund_ratio']:.1f} months")
    print(f"  Savings Rate: {health['savings_rate']:.1%}")
    print(f"  Debt-to-Income Ratio: {health['debt_to_income_ratio']:.1%}")
    print(f"  Investment Ratio: {health['investment_ratio']:.1%}")
    print(f"  Net Worth: ${health['net_worth']:,.0f}")
    
    print("\nRisk Factors:")
    for factor in summary['risk_assessment']['risk_factors']:
        print(f"  • {factor}")
    
    if summary['lending_recommendations']['conditions']:
        print("\nLoan Conditions:")
        for condition in summary['lending_recommendations']['conditions']:
            print(f"  • {condition}")
    
    if summary['lending_recommendations']['risk_mitigation']:
        print("\nRisk Mitigation:")
        for mitigation in summary['lending_recommendations']['risk_mitigation']:
            print(f"  • {mitigation}")

def test_multiple_customers():
    """Test analysis on multiple customers"""
    print("\n🧪 Testing Multiple Customer Analysis")
    print("=" * 50)
    
    # Get all sample customers
    customers = get_all_sample_customers()
    
    # Initialize analyzer
    analyzer = CustomerFinancialAnalyzer()
    
    # Analyze all customers
    results = []
    for customer in customers:
        summary = analyzer.create_customer_summary(customer)
        results.append(summary)
    
    # Calculate statistics
    total_customers = len(results)
    approved_count = sum(1 for r in results if r['lending_recommendations']['loan_approval'])
    avg_credit_score = sum(r['credit_score'] for r in results) / total_customers
    
    # Risk distribution
    risk_counts = {}
    for result in results:
        risk_level = result['risk_assessment']['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
    
    # Display summary
    print(f"Total Customers Analyzed: {total_customers}")
    print(f"Loan Approvals: {approved_count}/{total_customers} ({approved_count/total_customers:.1%})")
    print(f"Average Credit Score: {avg_credit_score:.0f}")
    
    print("\nRisk Level Distribution:")
    for risk_level, count in risk_counts.items():
        print(f"  {risk_level}: {count} customers ({count/total_customers:.1%})")
    
    # Show top 3 customers by credit score
    print("\nTop 3 Customers by Credit Score:")
    sorted_results = sorted(results, key=lambda x: x['credit_score'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result['customer_name']}: {result['credit_score']} ({result['credit_rating']})")
    
    # Show customers with highest risk
    print("\nHigh Risk Customers:")
    high_risk = [r for r in results if r['risk_assessment']['risk_level'] in ['High', 'Very High']]
    for result in high_risk:
        print(f"  • {result['customer_name']}: {result['risk_assessment']['risk_level']} Risk")

def test_credit_score_calculation():
    """Test credit score calculation with different scenarios"""
    print("\n🧪 Testing Credit Score Calculation")
    print("=" * 50)
    
    analyzer = CustomerFinancialAnalyzer()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Excellent Customer',
            'data': {
                'monthly_income': 12000,
                'monthly_expenses': 6000,
                'savings_balance': 50000,
                'investment_balance': 100000,
                'total_debt': 200000,
                'payment_history_score': 0.98,
                'credit_utilization_ratio': 0.15,
                'credit_age_months': 120
            }
        },
        {
            'name': 'Good Customer',
            'data': {
                'monthly_income': 8000,
                'monthly_expenses': 5000,
                'savings_balance': 20000,
                'investment_balance': 40000,
                'total_debt': 150000,
                'payment_history_score': 0.90,
                'credit_utilization_ratio': 0.30,
                'credit_age_months': 84
            }
        },
        {
            'name': 'Fair Customer',
            'data': {
                'monthly_income': 5000,
                'monthly_expenses': 4000,
                'savings_balance': 5000,
                'investment_balance': 10000,
                'total_debt': 80000,
                'payment_history_score': 0.75,
                'credit_utilization_ratio': 0.50,
                'credit_age_months': 48
            }
        },
        {
            'name': 'Poor Customer',
            'data': {
                'monthly_income': 3000,
                'monthly_expenses': 2800,
                'savings_balance': 1000,
                'investment_balance': 2000,
                'total_debt': 40000,
                'payment_history_score': 0.60,
                'credit_utilization_ratio': 0.80,
                'credit_age_months': 24
            }
        }
    ]
    
    for scenario in scenarios:
        credit_score = analyzer.calculate_credit_score(scenario['data'])
        rating = analyzer._get_credit_rating(credit_score)
        print(f"{scenario['name']}: {credit_score} ({rating})")

def main():
    """Run all tests"""
    print("🏦 Banker's Financial Insights Model - Test Suite")
    print("=" * 60)
    
    try:
        test_single_customer()
        test_multiple_customers()
        test_credit_score_calculation()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Create sample CSV: python main.py --create-sample test_customers.csv")
        print("   2. Run analysis: python main.py --csv test_customers.csv")
        print("   3. Launch dashboard: streamlit run dashboard/financial_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 