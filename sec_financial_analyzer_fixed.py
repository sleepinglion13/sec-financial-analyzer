#!/usr/bin/env python3
"""
SEC Financial Model Automation Script - FIXED VERSION
===================================================

This version works with standard packages and doesn't require problematic dependencies.
Uses direct SEC API calls and standard financial libraries.

Features:
- Direct SEC EDGAR API access (no special packages needed)
- Financial statement extraction from SEC JSON data
- DCF models with customizable assumptions  
- Comprehensive ratio analysis
- Risk detection and alerts
- Works with standard Python packages

Requirements (all standard packages):
- pandas
- numpy 
- requests
- yfinance
- matplotlib (optional)
- datetime

Author: Fixed for reliability
License: MIT
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import warnings
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from urllib.parse import urljoin
import re

# Optional imports with graceful fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available - install with: pip install yfinance")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sec_financial_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SECFinancialAnalyzer:
    """
    SEC filings analyzer using direct API calls - no problematic dependencies.
    """
    
    def __init__(self, user_agent: str = "financial.analyst@example.com"):
        """
        Initialize the analyzer.
        
        Args:
            user_agent: Required user agent for SEC API calls
        """
        self.user_agent = user_agent
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # SEC API endpoints
        self.sec_api_base = "https://data.sec.gov"
        self.company_tickers_url = f"{self.sec_api_base}/files/company_tickers.json"
        
        # Rate limiting (SEC requires max 10 requests per second)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Financial mappings for data extraction
        self.financial_mappings = {
            'revenue': ['Revenues', 'Revenue', 'Net Sales', 'Sales', 'Total Revenue'],
            'cost_of_revenue': ['Cost of Revenue', 'Cost of Sales', 'Cost of Goods Sold'],
            'gross_profit': ['Gross Profit', 'Gross Income'],
            'operating_income': ['Operating Income', 'Operating Profit', 'Income from Operations'],
            'net_income': ['Net Income', 'Net Earnings', 'Net Income (Loss)'],
            'total_assets': ['Total Assets', 'Assets'],
            'total_liabilities': ['Total Liabilities', 'Liabilities'],
            'shareholders_equity': ['Total Stockholders Equity', 'Stockholders Equity', 'Total Equity'],
            'cash_operations': ['Net Cash Provided by Operating Activities', 'Cash from Operations'],
            'free_cash_flow': ['Free Cash Flow']
        }
        
        # Load company ticker mapping
        self.ticker_to_cik = self._load_ticker_mapping()
        
    def _rate_limit(self):
        """Enforce SEC rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _load_ticker_mapping(self) -> Dict[str, str]:
        """Load ticker to CIK mapping from SEC."""
        try:
            self._rate_limit()
            response = requests.get(self.company_tickers_url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            ticker_map = {}
            
            for entry in data.values():
                if 'ticker' in entry and 'cik_str' in entry:
                    ticker_map[entry['ticker'].upper()] = str(entry['cik_str']).zfill(10)
            
            logger.info(f"Loaded {len(ticker_map)} ticker mappings")
            return ticker_map
            
        except Exception as e:
            logger.error(f"Error loading ticker mapping: {e}")
            return {}
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker symbol."""
        ticker = ticker.upper()
        return self.ticker_to_cik.get(ticker)
    
    def fetch_company_facts(self, ticker: str) -> Dict[str, Any]:
        """Fetch company facts from SEC API."""
        cik = self.get_company_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker {ticker}")
            return {}
        
        try:
            self._rate_limit()
            facts_url = f"{self.sec_api_base}/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(facts_url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching company facts for {ticker}: {e}")
            return {}
    
    def fetch_recent_filings(self, ticker: str, days_back: int = 90) -> List[Dict]:
        """Fetch recent filings for a company."""
        cik = self.get_company_cik(ticker)
        if not cik:
            return []
        
        try:
            self._rate_limit()
            submissions_url = f"{self.sec_api_base}/submissions/CIK{cik}.json"
            response = requests.get(submissions_url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            if not filings:
                return []
            
            # Extract filing information
            recent_filings = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for i in range(len(filings.get('form', []))):
                filing_date_str = filings['filingDate'][i]
                filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
                
                if filing_date >= cutoff_date:
                    recent_filings.append({
                        'form': filings['form'][i],
                        'filingDate': filing_date_str,
                        'accessionNumber': filings['accessionNumber'][i],
                        'primaryDocument': filings['primaryDocument'][i],
                        'ticker': ticker
                    })
            
            return recent_filings[:10]  # Return 10 most recent
            
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    def extract_financial_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Extract financial data from SEC company facts."""
        logger.info(f"Extracting financial data for {ticker}")
        
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return {}
        
        financial_data = {}
        
        try:
            # Extract from US-GAAP facts
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            
            if us_gaap:
                # Process different statement types
                financial_data['income_statement'] = self._extract_statement_data(
                    us_gaap, 'income_statement'
                )
                financial_data['balance_sheet'] = self._extract_statement_data(
                    us_gaap, 'balance_sheet'
                )
                financial_data['cash_flow'] = self._extract_statement_data(
                    us_gaap, 'cash_flow'
                )
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")
            return {}
    
    def _extract_statement_data(self, us_gaap_data: Dict, statement_type: str) -> pd.DataFrame:
        """Extract specific financial statement data."""
        statement_data = []
        
        # Define key fields for each statement type
        field_mappings = {
            'income_statement': [
                'Revenues', 'Revenue', 'CostOfRevenue', 'GrossProfit',
                'OperatingIncomeLoss', 'NetIncomeLoss', 'EarningsPerShareBasic'
            ],
            'balance_sheet': [
                'Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent',
                'StockholdersEquity', 'Cash', 'CashAndCashEquivalentsAtCarryingValue'
            ],
            'cash_flow': [
                'NetCashProvidedByUsedInOperatingActivities',
                'NetCashProvidedByUsedInInvestingActivities',
                'NetCashProvidedByUsedInFinancingActivities'
            ]
        }
        
        relevant_fields = field_mappings.get(statement_type, [])
        
        for field in relevant_fields:
            if field in us_gaap_data:
                field_data = us_gaap_data[field]
                if 'units' in field_data and 'USD' in field_data['units']:
                    usd_data = field_data['units']['USD']
                    for item in usd_data:
                        if 'val' in item and 'end' in item:
                            statement_data.append({
                                'field': field,
                                'value': item['val'],
                                'date': item['end'],
                                'period': item.get('fy', 'Unknown')
                            })
        
        if statement_data:
            df = pd.DataFrame(statement_data)
            df['date'] = pd.to_datetime(df['date'])
            return df.pivot(index='date', columns='field', values='value').fillna(0)
        else:
            return pd.DataFrame()
    
    def calculate_financial_ratios(self, financial_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate key financial ratios."""
        ratios = {}
        
        try:
            income_stmt = financial_data.get('income_statement', pd.DataFrame())
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            
            if not income_stmt.empty and not balance_sheet.empty:
                # Get most recent data
                latest_income = income_stmt.iloc[-1] if len(income_stmt) > 0 else None
                latest_balance = balance_sheet.iloc[-1] if len(balance_sheet) > 0 else None
                
                if latest_income is not None and latest_balance is not None:
                    # Revenue and profit metrics
                    revenue = self._get_field_value(latest_income, ['Revenues', 'Revenue'])
                    net_income = self._get_field_value(latest_income, ['NetIncomeLoss'])
                    
                    # Balance sheet metrics
                    total_assets = self._get_field_value(latest_balance, ['Assets'])
                    total_equity = self._get_field_value(latest_balance, ['StockholdersEquity'])
                    current_assets = self._get_field_value(latest_balance, ['AssetsCurrent'])
                    current_liabilities = self._get_field_value(latest_balance, ['LiabilitiesCurrent'])
                    total_liabilities = self._get_field_value(latest_balance, ['Liabilities'])
                    
                    # Calculate ratios
                    if revenue and revenue != 0:
                        ratios['Net Profit Margin'] = (net_income / revenue) * 100 if net_income else 0
                    
                    if total_assets and total_assets != 0:
                        ratios['ROA'] = (net_income / total_assets) * 100 if net_income else 0
                    
                    if total_equity and total_equity != 0:
                        ratios['ROE'] = (net_income / total_equity) * 100 if net_income else 0
                    
                    if current_liabilities and current_liabilities != 0:
                        ratios['Current Ratio'] = current_assets / current_liabilities if current_assets else 0
                    
                    if total_equity and total_equity != 0:
                        ratios['Debt to Equity'] = total_liabilities / total_equity if total_liabilities else 0
                    
                    # Revenue growth (if multiple periods available)
                    if len(income_stmt) >= 2:
                        prev_revenue = self._get_field_value(income_stmt.iloc[-2], ['Revenues', 'Revenue'])
                        if prev_revenue and prev_revenue != 0:
                            ratios['Revenue Growth'] = ((revenue - prev_revenue) / prev_revenue) * 100
        
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
        
        return ratios
    
    def _get_field_value(self, data_row, field_names: List[str]) -> Optional[float]:
        """Get value from data row using multiple possible field names."""
        for field in field_names:
            if field in data_row.index and pd.notna(data_row[field]):
                return float(data_row[field])
        return None
    
    def build_simple_dcf(self, ticker: str, financial_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Build a simplified DCF model."""
        logger.info(f"Building DCF model for {ticker}")
        
        try:
            cash_flow = financial_data.get('cash_flow', pd.DataFrame())
            
            if cash_flow.empty:
                logger.warning("No cash flow data for DCF")
                return {}
            
            # Extract operating cash flows
            ocf_field = 'NetCashProvidedByUsedInOperatingActivities'
            if ocf_field not in cash_flow.columns:
                logger.warning("No operating cash flow data found")
                return {}
            
            # Get cash flow history
            cash_flows = cash_flow[ocf_field].dropna().tolist()
            
            if len(cash_flows) < 2:
                logger.warning("Insufficient cash flow history")
                return {}
            
            # Calculate average growth rate
            growth_rates = []
            for i in range(1, len(cash_flows)):
                if cash_flows[i-1] != 0:
                    growth = (cash_flows[i] - cash_flows[i-1]) / abs(cash_flows[i-1])
                    growth_rates.append(growth)
            
            avg_growth = np.mean(growth_rates) if growth_rates else 0.05
            
            # DCF parameters
            projection_years = 5
            terminal_growth = 0.025
            discount_rate = 0.10
            
            # Project future cash flows
            latest_cf = cash_flows[-1]
            projected_cf = []
            
            for year in range(1, projection_years + 1):
                cf = latest_cf * ((1 + avg_growth) ** year)
                projected_cf.append(cf)
            
            # Terminal value
            terminal_cf = projected_cf[-1] * (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            
            # Present values
            pv_cf = []
            for year, cf in enumerate(projected_cf, 1):
                pv = cf / ((1 + discount_rate) ** year)
                pv_cf.append(pv)
            
            pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
            enterprise_value = sum(pv_cf) + pv_terminal
            
            # Get shares outstanding from market data if available
            shares_outstanding = self._get_shares_outstanding(ticker)
            price_per_share = enterprise_value / shares_outstanding if shares_outstanding else None
            
            return {
                'ticker': ticker,
                'latest_operating_cf': latest_cf,
                'avg_growth_rate': avg_growth,
                'projected_cash_flows': projected_cf,
                'terminal_value': terminal_value,
                'enterprise_value': enterprise_value,
                'shares_outstanding': shares_outstanding,
                'dcf_price_per_share': price_per_share,
                'discount_rate': discount_rate,
                'terminal_growth_rate': terminal_growth
            }
            
        except Exception as e:
            logger.error(f"Error building DCF model: {e}")
            return {}
    
    def _get_shares_outstanding(self, ticker: str) -> Optional[float]:
        """Get shares outstanding from yfinance if available."""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('sharesOutstanding')
        except:
            return None
    
    def detect_financial_alerts(self, ratios: Dict[str, float]) -> List[str]:
        """Detect financial red flags."""
        alerts = []
        
        try:
            # Profitability alerts
            if ratios.get('Net Profit Margin', 0) < 0:
                alerts.append("⚠️ ALERT: Negative profit margin")
            
            # Liquidity alerts
            current_ratio = ratios.get('Current Ratio', 0)
            if current_ratio > 0 and current_ratio < 1.0:
                alerts.append("⚠️ ALERT: Current ratio below 1.0 - liquidity concerns")
            
            # Leverage alerts
            debt_equity = ratios.get('Debt to Equity', 0)
            if debt_equity > 2.0:
                alerts.append("⚠️ ALERT: High debt-to-equity ratio - leverage risk")
            
            # Growth alerts
            revenue_growth = ratios.get('Revenue Growth')
            if revenue_growth is not None and revenue_growth < -10:
                alerts.append("⚠️ ALERT: Significant revenue decline detected")
        
        except Exception as e:
            logger.error(f"Error detecting alerts: {e}")
        
        return alerts
    
    def generate_comprehensive_report(self, ticker: str) -> Dict[str, Any]:
        """Generate complete financial analysis report."""
        logger.info(f"Generating report for {ticker}")
        
        report = {
            'ticker': ticker.upper(),
            'analysis_date': datetime.now().isoformat(),
            'filings': [],
            'financial_data_available': False,
            'ratios': {},
            'dcf_model': {},
            'alerts': [],
            'summary': {}
        }
        
        try:
            # 1. Get recent filings
            report['filings'] = self.fetch_recent_filings(ticker)
            
            # 2. Extract financial data
            financial_data = self.extract_financial_data(ticker)
            report['financial_data_available'] = bool(financial_data)
            
            # 3. Calculate ratios
            if financial_data:
                report['ratios'] = self.calculate_financial_ratios(financial_data)
            
            # 4. Build DCF model
            if financial_data:
                report['dcf_model'] = self.build_simple_dcf(ticker, financial_data)
            
            # 5. Detect alerts
            if report['ratios']:
                report['alerts'] = self.detect_financial_alerts(report['ratios'])
            
            # 6. Generate summary
            report['summary'] = {
                'filings_found': len(report['filings']),
                'data_quality': 'Good' if financial_data else 'Limited',
                'dcf_available': bool(report['dcf_model']),
                'alerts_count': len(report['alerts'])
            }
            
            # Add key ratios to summary
            if report['ratios']:
                key_ratios = {k: v for k, v in report['ratios'].items() 
                            if k in ['Net Profit Margin', 'ROE', 'Current Ratio', 'Debt to Equity']}
                report['summary']['key_ratios'] = key_ratios
            
            # Add DCF valuation to summary
            if report['dcf_model'] and 'dcf_price_per_share' in report['dcf_model']:
                report['summary']['dcf_valuation'] = report['dcf_model']['dcf_price_per_share']
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            report['error'] = str(e)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_dir: str = ".") -> str:
        """Save report to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/financial_analysis_{report['ticker']}_{timestamp}.json"
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SEC Financial Analysis - Fixed Version')
    parser.add_argument('ticker', help='Company ticker symbol (e.g., AAPL)')
    parser.add_argument('--output-dir', default='reports', help='Output directory')
    parser.add_argument('--days-back', type=int, default=90, help='Days back for filings')
    parser.add_argument('--user-agent', default='financial.analyst@example.com', 
                       help='User agent for SEC API')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SECFinancialAnalyzer(user_agent=args.user_agent)
    
    print(f"\n🔍 Analyzing {args.ticker}...")
    print("📋 Using direct SEC API calls (no problematic packages)")
    
    # Generate report
    report = analyzer.generate_comprehensive_report(args.ticker)
    
    # Display summary
    print(f"\n📊 Analysis Summary for {args.ticker}")
    print("=" * 50)
    
    summary = report.get('summary', {})
    print(f"Recent Filings: {summary.get('filings_found', 0)}")
    print(f"Data Quality: {summary.get('data_quality', 'Unknown')}")
    print(f"DCF Available: {summary.get('dcf_available', False)}")
    print(f"Alerts: {summary.get('alerts_count', 0)}")
    
    # Show ratios
    if 'key_ratios' in summary:
        print("\nKey Ratios:")
        for ratio, value in summary['key_ratios'].items():
            if value is not None:
                unit = '%' if 'Margin' in ratio or 'RO' in ratio or 'Growth' in ratio else ''
                print(f"  {ratio}: {value:.2f}{unit}")
    
    # Show DCF
    if 'dcf_valuation' in summary and summary['dcf_valuation']:
        print(f"\nDCF Valuation: ${summary['dcf_valuation']:.2f}")
    
    # Show alerts
    if report['alerts']:
        print("\n🚨 Alerts:")
        for alert in report['alerts']:
            print(f"  {alert}")
    
    # Save report
    filename = analyzer.save_report(report, args.output_dir)
    if filename:
        print(f"\n✅ Report saved: {filename}")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()