"""
Cohort Analysis Module for Sportsbook Marketing Analytics

This module provides comprehensive cohort analysis functionality including
retention curves, revenue cohorts, and customer lifecycle analysis.

Author: Sportsbook Marketing Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class CohortAnalyzer:
    """
    Comprehensive cohort analysis for customer retention and revenue patterns
    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the cohort analyzer
        
        Args:
            data_path: Path to the raw data files
        """
        self.data_path = data_path
        self.customers_df = None
        self.transactions_df = None
        self.betting_df = None
        
    def load_data(self):
        """Load customer, transaction, and betting data"""
        try:
            print("Loading data for cohort analysis...")
            
            self.customers_df = pd.read_csv(f"{self.data_path}/customers.csv")
            self.customers_df['signup_date'] = pd.to_datetime(self.customers_df['signup_date'])
            
            self.transactions_df = pd.read_csv(f"{self.data_path}/transactions.csv")
            self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
            
            self.betting_df = pd.read_csv(f"{self.data_path}/betting_activity.csv")
            self.betting_df['bet_date'] = pd.to_datetime(self.betting_df['bet_date'])
            
            print(f"  Loaded {len(self.customers_df):,} customers")
            print(f"  Loaded {len(self.transactions_df):,} transactions")
            print(f"  Loaded {len(self.betting_df):,} bets")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run the data generator first: python python/data_generator.py")
            
    def create_retention_cohorts(self, period: str = 'monthly') -> pd.DataFrame:
        """
        Create retention cohort analysis
        
        Args:
            period: 'monthly' or 'weekly' cohort grouping
            
        Returns:
            DataFrame with cohort retention data
        """
        if self.customers_df is None or self.betting_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Creating {period} retention cohorts...")
        
        # Determine period grouping
        if period == 'monthly':
            period_group = 'M'
            date_format = '%Y-%m'
        else:
            period_group = 'W'
            date_format = '%Y-%W'
        
        # Create cohort groups based on signup month/week
        cohort_data = self.customers_df.copy()
        cohort_data['cohort_period'] = cohort_data['signup_date'].dt.to_period(period_group)
        
        # Get all customer activities (bets and transactions)
        activities = []
        
        # Add betting activities
        betting_activities = self.betting_df[['customer_id', 'bet_date']].copy()
        betting_activities['activity_date'] = betting_activities['bet_date']
        betting_activities = betting_activities[['customer_id', 'activity_date']]
        activities.append(betting_activities)
        
        # Add transaction activities  
        transaction_activities = self.transactions_df[['customer_id', 'transaction_date']].copy()
        transaction_activities['activity_date'] = transaction_activities['transaction_date']
        transaction_activities = transaction_activities[['customer_id', 'activity_date']]
        activities.append(transaction_activities)
        
        # Combine all activities
        all_activities = pd.concat(activities, ignore_index=True)
        all_activities['activity_period'] = all_activities['activity_date'].dt.to_period(period_group)
        
        # Join cohort info with activities
        cohort_activities = cohort_data[['customer_id', 'cohort_period']].merge(
            all_activities, on='customer_id', how='left'
        )
        
        # Calculate period number (0 = signup period, 1 = first period after signup, etc.)
        # Handle NaT values properly
        def safe_period_diff(row):
            try:
                if pd.isna(row['activity_period']) or pd.isna(row['cohort_period']):
                    return None
                diff = row['activity_period'] - row['cohort_period']
                return diff.n if hasattr(diff, 'n') else None
            except:
                return None
        
        cohort_activities['period_number'] = cohort_activities.apply(safe_period_diff, axis=1)
        
        # Remove rows with None period numbers (NaT values)
        cohort_activities = cohort_activities.dropna(subset=['period_number'])
        cohort_activities['period_number'] = cohort_activities['period_number'].astype(int)
        
        # Remove negative periods (activities before signup - shouldn't happen but just in case)
        cohort_activities = cohort_activities[cohort_activities['period_number'] >= 0]
        
        # Create cohort table
        cohort_table = cohort_activities.groupby(['cohort_period', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_table.rename(columns={'customer_id': 'active_customers'}, inplace=True)
        
        # Get cohort sizes (number of customers in each cohort)
        cohort_sizes = cohort_data.groupby('cohort_period')['customer_id'].nunique().reset_index()
        cohort_sizes.rename(columns={'customer_id': 'cohort_size'}, inplace=True)
        
        # Calculate retention rates
        cohort_table = cohort_table.merge(cohort_sizes, on='cohort_period')
        cohort_table['retention_rate'] = cohort_table['active_customers'] / cohort_table['cohort_size']
        
        # Pivot to create retention matrix
        retention_matrix = cohort_table.pivot(
            index='cohort_period', 
            columns='period_number', 
            values='retention_rate'
        ).fillna(0)
        
        return retention_matrix
    
    def create_revenue_cohorts(self, period: str = 'monthly') -> pd.DataFrame:
        """
        Create revenue-based cohort analysis
        
        Args:
            period: 'monthly' or 'weekly' cohort grouping
            
        Returns:
            DataFrame with cohort revenue data
        """
        if self.customers_df is None or self.betting_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Creating {period} revenue cohorts...")
        
        # Determine period grouping
        if period == 'monthly':
            period_group = 'M'
        else:
            period_group = 'W'
        
        # Create cohort groups
        cohort_data = self.customers_df.copy()
        cohort_data['cohort_period'] = cohort_data['signup_date'].dt.to_period(period_group)
        
        # Calculate customer revenue (gross gaming revenue)
        betting_revenue = self.betting_df.copy()
        betting_revenue['revenue'] = betting_revenue['bet_amount'] - betting_revenue['payout']
        betting_revenue['activity_period'] = betting_revenue['bet_date'].dt.to_period(period_group)
        
        # Group by customer and period
        customer_revenue = betting_revenue.groupby(['customer_id', 'activity_period'])['revenue'].sum().reset_index()
        
        # Join with cohort data
        cohort_revenue = cohort_data[['customer_id', 'cohort_period']].merge(
            customer_revenue, on='customer_id', how='left'
        )
        
        # Calculate period number
        cohort_revenue['period_number'] = (
            cohort_revenue['activity_period'] - cohort_revenue['cohort_period']
        ).apply(attrgetter('n'))
        
        # Remove negative periods
        cohort_revenue = cohort_revenue[cohort_revenue['period_number'] >= 0]
        
        # Aggregate revenue by cohort and period
        revenue_cohort_table = cohort_revenue.groupby(['cohort_period', 'period_number']).agg({
            'revenue': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        revenue_cohort_table.columns = [
            'cohort_period', 'period_number', 'total_revenue', 
            'avg_revenue_per_active', 'revenue_events', 'active_customers'
        ]
        
        # Get cohort sizes
        cohort_sizes = cohort_data.groupby('cohort_period')['customer_id'].nunique().reset_index()
        cohort_sizes.rename(columns={'customer_id': 'cohort_size'}, inplace=True)
        
        # Calculate revenue per customer in cohort
        revenue_cohort_table = revenue_cohort_table.merge(cohort_sizes, on='cohort_period')
        revenue_cohort_table['revenue_per_cohort_customer'] = (
            revenue_cohort_table['total_revenue'] / revenue_cohort_table['cohort_size']
        )
        
        return revenue_cohort_table
    
    def calculate_customer_lifetime_value_cohorts(self) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value by cohorts
        
        Returns:
            DataFrame with LTV analysis by cohort
        """
        if self.customers_df is None or self.betting_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Calculating Customer Lifetime Value by cohorts...")
        
        # Create monthly cohorts
        cohort_data = self.customers_df.copy()
        cohort_data['cohort_month'] = cohort_data['signup_date'].dt.to_period('M')
        
        # Calculate customer-level metrics
        customer_metrics = []
        
        for _, customer in cohort_data.iterrows():
            customer_id = customer['customer_id']
            signup_date = customer['signup_date']
            cohort_month = customer['cohort_month']
            
            # Get customer's betting activity
            customer_bets = self.betting_df[self.betting_df['customer_id'] == customer_id].copy()
            
            if len(customer_bets) > 0:
                # Calculate revenue (house perspective)
                total_revenue = (customer_bets['bet_amount'] - customer_bets['payout']).sum()
                
                # Calculate tenure
                first_bet_date = customer_bets['bet_date'].min()
                last_bet_date = customer_bets['bet_date'].max()
                tenure_days = (last_bet_date - first_bet_date).days + 1
                
                # Calculate other metrics
                total_bets = len(customer_bets)
                avg_bet_size = customer_bets['bet_amount'].mean()
                betting_frequency = total_bets / tenure_days if tenure_days > 0 else 0
                
            else:
                total_revenue = 0
                tenure_days = 0
                total_bets = 0
                avg_bet_size = 0
                betting_frequency = 0
            
            # Get deposits
            customer_deposits = self.transactions_df[
                (self.transactions_df['customer_id'] == customer_id) &
                (self.transactions_df['transaction_type'] == 'deposit')
            ]
            total_deposits = customer_deposits['amount'].sum() if len(customer_deposits) > 0 else 0
            
            # Calculate LTV (revenue minus acquisition cost)
            acquisition_cost = customer['acquisition_cost']
            ltv = total_revenue - acquisition_cost
            
            customer_metrics.append({
                'customer_id': customer_id,
                'cohort_month': cohort_month,
                'acquisition_channel': customer['acquisition_channel'],
                'customer_segment': customer['customer_segment'],
                'acquisition_cost': acquisition_cost,
                'total_revenue': total_revenue,
                'total_deposits': total_deposits,
                'ltv': ltv,
                'tenure_days': tenure_days,
                'total_bets': total_bets,
                'avg_bet_size': avg_bet_size,
                'betting_frequency': betting_frequency
            })
        
        customer_ltv_df = pd.DataFrame(customer_metrics)
        
        # Aggregate by cohort
        cohort_ltv_summary = customer_ltv_df.groupby('cohort_month').agg({
            'customer_id': 'count',
            'ltv': ['mean', 'median', 'std', 'sum'],
            'total_revenue': ['mean', 'median', 'sum'],
            'total_deposits': ['mean', 'sum'],
            'acquisition_cost': 'mean',
            'tenure_days': 'mean',
            'total_bets': 'mean',
            'avg_bet_size': 'mean',
            'betting_frequency': 'mean'
        }).round(2)
        
        # Flatten column names
        cohort_ltv_summary.columns = [
            'customer_count', 'avg_ltv', 'median_ltv', 'std_ltv', 'total_ltv',
            'avg_revenue', 'median_revenue', 'total_revenue',
            'avg_deposits', 'total_deposits', 'avg_acquisition_cost',
            'avg_tenure_days', 'avg_total_bets', 'avg_bet_size', 'avg_betting_frequency'
        ]
        
        # Calculate additional metrics
        cohort_ltv_summary['ltv_cac_ratio'] = cohort_ltv_summary['avg_ltv'] / cohort_ltv_summary['avg_acquisition_cost']
        cohort_ltv_summary['revenue_per_customer'] = cohort_ltv_summary['total_revenue'] / cohort_ltv_summary['customer_count']
        
        return cohort_ltv_summary.reset_index()
    
    def analyze_churn_by_cohort(self, churn_period_days: int = 30) -> pd.DataFrame:
        """
        Analyze churn rates by cohort
        
        Args:
            churn_period_days: Number of days of inactivity to consider churned
            
        Returns:
            DataFrame with churn analysis by cohort
        """
        if self.customers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Analyzing churn by cohort (churn threshold: {churn_period_days} days)...")
        
        # Create cohort groups
        cohort_data = self.customers_df.copy()
        cohort_data['cohort_month'] = cohort_data['signup_date'].dt.to_period('M')
        
        # Get last activity date for each customer
        customer_last_activity = []
        
        for customer_id in cohort_data['customer_id']:
            # Find last betting activity
            customer_bets = self.betting_df[self.betting_df['customer_id'] == customer_id]
            last_bet_date = customer_bets['bet_date'].max() if len(customer_bets) > 0 else None
            
            # Find last transaction
            customer_transactions = self.transactions_df[self.transactions_df['customer_id'] == customer_id]
            last_transaction_date = customer_transactions['transaction_date'].max() if len(customer_transactions) > 0 else None
            
            # Get the latest activity date
            activity_dates = [d for d in [last_bet_date, last_transaction_date] if pd.notna(d)]
            last_activity_date = max(activity_dates) if activity_dates else None
            
            customer_last_activity.append({
                'customer_id': customer_id,
                'last_activity_date': last_activity_date
            })
        
        last_activity_df = pd.DataFrame(customer_last_activity)
        
        # Join with cohort data
        cohort_churn_data = cohort_data.merge(last_activity_df, on='customer_id')
        
        # Calculate days since last activity
        current_date = pd.Timestamp.now().normalize()
        cohort_churn_data['days_since_last_activity'] = (
            current_date - cohort_churn_data['last_activity_date']
        ).dt.days
        
        # Define churn status
        cohort_churn_data['is_churned'] = (
            (cohort_churn_data['days_since_last_activity'] > churn_period_days) |
            cohort_churn_data['last_activity_date'].isna()
        )
        
        # Never active customers (signed up but never did anything)
        cohort_churn_data['never_active'] = cohort_churn_data['last_activity_date'].isna()
        
        # Aggregate by cohort
        churn_summary = cohort_churn_data.groupby('cohort_month').agg({
            'customer_id': 'count',
            'is_churned': ['sum', 'mean'],
            'never_active': ['sum', 'mean'],
            'days_since_last_activity': ['mean', 'median']
        }).round(4)
        
        # Flatten column names
        churn_summary.columns = [
            'total_customers', 'churned_customers', 'churn_rate',
            'never_active_customers', 'never_active_rate',
            'avg_days_since_activity', 'median_days_since_activity'
        ]
        
        # Calculate active customers
        churn_summary['active_customers'] = churn_summary['total_customers'] - churn_summary['churned_customers']
        churn_summary['active_rate'] = 1 - churn_summary['churn_rate']
        
        return churn_summary.reset_index()
        
    def plot_retention_heatmap(self, retention_matrix: pd.DataFrame, title: str = "Customer Retention Heatmap"):
        """
        Plot retention cohort heatmap
        
        Args:
            retention_matrix: Output from create_retention_cohorts()
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Convert period index to string for better display
        retention_display = retention_matrix.copy()
        retention_display.index = retention_display.index.astype(str)
        
        sns.heatmap(
            retention_display,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            cbar_kws={'label': 'Retention Rate'},
            linewidths=0.5
        )
        
        plt.title(title)
        plt.xlabel('Period Number (0 = Signup Period)')
        plt.ylabel('Cohort Period')
        plt.tight_layout()
        plt.show()
    
    def plot_revenue_cohorts(self, revenue_cohorts: pd.DataFrame):
        """
        Plot revenue cohort analysis
        
        Args:
            revenue_cohorts: Output from create_revenue_cohorts()
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Revenue per cohort customer over time
        pivot_revenue = revenue_cohorts.pivot(
            index='cohort_period',
            columns='period_number', 
            values='revenue_per_cohort_customer'
        ).fillna(0)
        
        pivot_revenue.index = pivot_revenue.index.astype(str)
        
        sns.heatmap(
            pivot_revenue,
            annot=True,
            fmt='.1f',
            cmap='Greens',
            ax=axes[0, 0],
            cbar_kws={'label': 'Revenue per Customer ($)'}
        )
        axes[0, 0].set_title('Revenue per Cohort Customer')
        axes[0, 0].set_xlabel('Period Number')
        axes[0, 0].set_ylabel('Cohort Period')
        
        # 2. Total revenue by cohort over time
        pivot_total = revenue_cohorts.pivot(
            index='cohort_period',
            columns='period_number',
            values='total_revenue'
        ).fillna(0)
        
        pivot_total.index = pivot_total.index.astype(str)
        
        sns.heatmap(
            pivot_total,
            annot=True,
            fmt='.0f',
            cmap='Blues',
            ax=axes[0, 1],
            cbar_kws={'label': 'Total Revenue ($)'}
        )
        axes[0, 1].set_title('Total Revenue by Cohort')
        axes[0, 1].set_xlabel('Period Number')
        axes[0, 1].set_ylabel('Cohort Period')
        
        # 3. Average revenue per active customer
        pivot_avg_active = revenue_cohorts.pivot(
            index='cohort_period',
            columns='period_number',
            values='avg_revenue_per_active'
        ).fillna(0)
        
        pivot_avg_active.index = pivot_avg_active.index.astype(str)
        
        sns.heatmap(
            pivot_avg_active,
            annot=True,
            fmt='.1f',
            cmap='Purples',
            ax=axes[1, 0],
            cbar_kws={'label': 'Avg Revenue per Active ($)'}
        )
        axes[1, 0].set_title('Average Revenue per Active Customer')
        axes[1, 0].set_xlabel('Period Number')
        axes[1, 0].set_ylabel('Cohort Period')
        
        # 4. Number of active customers
        pivot_active = revenue_cohorts.pivot(
            index='cohort_period',
            columns='period_number',
            values='active_customers'
        ).fillna(0)
        
        pivot_active.index = pivot_active.index.astype(str)
        
        sns.heatmap(
            pivot_active,
            annot=True,
            fmt='.0f',
            cmap='Oranges',
            ax=axes[1, 1],
            cbar_kws={'label': 'Active Customers'}
        )
        axes[1, 1].set_title('Active Customers by Cohort')
        axes[1, 1].set_xlabel('Period Number')
        axes[1, 1].set_ylabel('Cohort Period')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_cohort_dashboard(self, retention_matrix: pd.DataFrame) -> go.Figure:
        """
        Create interactive cohort dashboard using Plotly
        
        Args:
            retention_matrix: Output from create_retention_cohorts()
            
        Returns:
            Plotly figure object
        """
        # Convert to format suitable for plotly
        retention_df = retention_matrix.reset_index()
        retention_df['cohort_period'] = retention_df['cohort_period'].astype(str)
        
        # Melt the dataframe for plotly
        retention_melted = retention_df.melt(
            id_vars=['cohort_period'],
            var_name='period_number',
            value_name='retention_rate'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=retention_matrix.values,
            x=retention_matrix.columns,
            y=[str(idx) for idx in retention_matrix.index],
            colorscale='YlOrRd',
            text=retention_matrix.values,
            texttemplate="%{text:.1%}",
            textfont={"size": 10},
            colorbar=dict(title="Retention Rate", tickformat=".1%")
        ))
        
        fig.update_layout(
            title="Customer Retention Cohort Analysis",
            xaxis_title="Period Number (0 = Signup Period)",
            yaxis_title="Cohort Period",
            height=600,
            width=1000
        )
        
        return fig


def main():
    """Example usage of cohort analysis"""
    print("📊 Sportsbook Cohort Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CohortAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    if analyzer.customers_df is None:
        print("❌ Data not available. Please run data generator first:")
        print("   python python/data_generator.py")
        return
    
    print("\n🔄 Running Cohort Analysis...")
    
    # 1. Retention cohort analysis
    print("\n1. Customer Retention Analysis")
    retention_matrix = analyzer.create_retention_cohorts('monthly')
    print(f"   Created retention matrix: {retention_matrix.shape}")
    print("\n   Monthly Retention Rates (first 6 months):")
    print(retention_matrix.iloc[:, :6].to_string())
    
    # 2. Revenue cohort analysis
    print("\n2. Revenue Cohort Analysis")
    revenue_cohorts = analyzer.create_revenue_cohorts('monthly')
    print(f"   Analyzed {len(revenue_cohorts)} cohort-period combinations")
    
    # Show sample of revenue cohorts
    sample_revenue = revenue_cohorts[revenue_cohorts['period_number'] <= 3]
    print("\n   Sample Revenue Cohorts (first 3 periods):")
    print(sample_revenue[['cohort_period', 'period_number', 'total_revenue', 
                         'revenue_per_cohort_customer', 'active_customers']].head(10).to_string(index=False))
    
    # 3. LTV cohort analysis
    print("\n3. Customer Lifetime Value Analysis")
    ltv_cohorts = analyzer.calculate_customer_lifetime_value_cohorts()
    print(f"   Analyzed {len(ltv_cohorts)} monthly cohorts")
    
    print("\n   LTV Summary by Cohort:")
    ltv_display = ltv_cohorts[['cohort_month', 'customer_count', 'avg_ltv', 
                              'avg_revenue', 'ltv_cac_ratio', 'avg_tenure_days']].head(8)
    print(ltv_display.to_string(index=False))
    
    # 4. Churn analysis
    print("\n4. Churn Analysis by Cohort")
    churn_analysis = analyzer.analyze_churn_by_cohort(churn_period_days=30)
    print(f"   Analyzed churn for {len(churn_analysis)} cohorts")
    
    churn_display = churn_analysis[['cohort_month', 'total_customers', 'churn_rate', 
                                   'never_active_rate', 'active_rate']].head(8)
    print("\n   Churn Rates by Cohort:")
    print(churn_display.to_string(index=False))
    
    # 5. Generate visualizations
    print("\n📈 Generating Visualizations...")
    
    # Retention heatmap
    analyzer.plot_retention_heatmap(retention_matrix)
    
    # Revenue cohort plots
    analyzer.plot_revenue_cohorts(revenue_cohorts)
    
    # Calculate key insights
    print("\n🔍 Key Insights:")
    
    # Overall retention rates
    period_1_retention = retention_matrix[1].mean() if 1 in retention_matrix.columns else 0
    period_3_retention = retention_matrix[3].mean() if 3 in retention_matrix.columns else 0
    period_6_retention = retention_matrix[6].mean() if 6 in retention_matrix.columns else 0
    
    print(f"   Average 1-month retention: {period_1_retention:.1%}")
    print(f"   Average 3-month retention: {period_3_retention:.1%}")
    print(f"   Average 6-month retention: {period_6_retention:.1%}")
    
    # LTV insights
    avg_ltv = ltv_cohorts['avg_ltv'].mean()
    avg_cac = ltv_cohorts['avg_acquisition_cost'].mean()
    avg_ltv_cac_ratio = avg_ltv / avg_cac if avg_cac > 0 else 0
    
    print(f"   Average Customer LTV: ${avg_ltv:.2f}")
    print(f"   Average Acquisition Cost: ${avg_cac:.2f}")
    print(f"   Overall LTV:CAC Ratio: {avg_ltv_cac_ratio:.1f}")
    
    # Churn insights
    overall_churn_rate = churn_analysis['churn_rate'].mean()
    never_active_rate = churn_analysis['never_active_rate'].mean()
    
    print(f"   Overall Churn Rate: {overall_churn_rate:.1%}")
    print(f"   Never Active Rate: {never_active_rate:.1%}")
    
    print("\n✅ Cohort analysis complete!")
    print("\nNext steps:")
    print("  1. Use insights to optimize marketing spend allocation")
    print("  2. Implement retention campaigns for at-risk cohorts")
    print("  3. Focus acquisition on high-LTV channels and segments")

if __name__ == "__main__":
    main()