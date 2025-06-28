import pandas as pd
import numpy as np
import re
from datetime import datetime

class BankStatementParser:
    """
    Universal parser for various Indian bank statement CSV formats.
    Extracts standardized features for financial analysis.
    """
    def __init__(self, customer_id='AUTO', customer_name='Unknown'):
        self.customer_id = customer_id
        self.customer_name = customer_name

    def parse(self, file_path):
        df = pd.read_csv(file_path)
        # Try to detect the format
        if set(['Date', 'Instrument ID', 'Amount', 'Type', 'Balance', 'Remarks']).issubset(df.columns):
            return self._parse_pnb(df)
        elif set(['Date', 'Details', 'Ref No./Cheque No', 'Debit', 'Credit', 'Balance']).issubset(df.columns):
            return self._parse_sbi(df)
        elif set(['Post Date', 'Value Date', 'Narration', 'Cheque Details', 'Debit', 'Credit', 'Balance']).issubset(df.columns):
            return self._parse_apgb(df)
        elif set(['Date', 'Description', 'Amount', 'Type']).issubset(df.columns):
            return self._parse_icici(df)
        else:
            raise ValueError('Unknown bank statement format. Please provide a supported CSV.')

    def _parse_pnb(self, df):
        # PNB: Date, Instrument ID, Amount, Type (DR/CR), Balance, Remarks
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['is_credit'] = df['Type'].str.upper().str.contains('CR')
        df['is_debit'] = df['Type'].str.upper().str.contains('DR')
        
        monthly_income = df[df['is_credit']]['Amount'].sum()
        monthly_expenses = df[df['is_debit']]['Amount'].sum()
        savings_balance = df['Balance'].apply(pd.to_numeric, errors='coerce').dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_sbi(self, df):
        # SBI: Date, Details, Ref No./Cheque No, Debit, Credit, Balance
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        
        monthly_income = df['Credit'].sum()
        monthly_expenses = df['Debit'].sum()
        savings_balance = df['Balance'].apply(pd.to_numeric, errors='coerce').dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_apgb(self, df):
        # APGB: Post Date, Value Date, Narration, Cheque Details, Debit, Credit, Balance
        df['Post Date'] = pd.to_datetime(df['Post Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Post Date'])
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        # Balance may have Dr/Cr suffix
        df['Balance'] = df['Balance'].astype(str).str.replace('Dr|Cr', '', regex=True)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
        
        monthly_income = df['Credit'].sum()
        monthly_expenses = df['Debit'].sum()
        savings_balance = df['Balance'].dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Post Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_icici(self, df):
        # ICICI: Date, Description, Amount, Type (DR/CR)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['is_credit'] = df['Type'].str.upper().str.contains('CR')
        df['is_debit'] = df['Type'].str.upper().str.contains('DR')
        
        monthly_income = df[df['is_credit']]['Amount'].sum()
        monthly_expenses = df[df['is_debit']]['Amount'].sum()
        # No balance column, so estimate from last transaction
        savings_balance = df[df['is_credit'] | df['is_debit']]['Amount'].cumsum().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _estimate_credit_age(self, date_series):
        if len(date_series) == 0:
            return 0
        min_date = date_series.min()
        max_date = date_series.max()
        months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        return max(1, months)

    def _standardize(self, monthly_income, monthly_expenses, savings_balance, credit_age_months):
        # Fill in defaults for missing fields
        return {
            'customer_id': self.customer_id,
            'customer_name': self.customer_name,
            'monthly_income': float(monthly_income),
            'monthly_expenses': float(monthly_expenses),
            'savings_balance': float(savings_balance),
            'investment_balance': 0.0,
            'total_debt': 0.0,
            'payment_history_score': 1.0,
            'credit_utilization_ratio': 0.0,
            'credit_age_months': int(credit_age_months)
        } 