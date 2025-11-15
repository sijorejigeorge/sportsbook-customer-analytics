"""
Streamlit Real-Time Marketing Analytics Dashboard

This module creates an interactive dashboard for sportsbook marketing analytics
including KPIs, cohort analysis, A/B testing results, and ML predictions.

Author: Sportsbook Marketing Analytics Team
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import os
import sys

# Add the python directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from cohort_analysis import CohortAnalyzer
    from ab_testing_framework import ABTestAnalyzer, CampaignSimulator
    from marketing_attribution import MarketingAttributionAnalyzer
    from ml_predictive_models import SportsbookMLPredictor
except ImportError:
    st.error("Could not import required modules. Please ensure all Python files are in the correct location.")

# Page config
st.set_page_config(
    page_title="Sportsbook Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        data_path = "data/raw"
        
        customers_df = pd.read_csv(f"{data_path}/customers.csv")
        customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
        
        transactions_df = pd.read_csv(f"{data_path}/transactions.csv")
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        
        betting_df = pd.read_csv(f"{data_path}/betting_activity.csv")
        betting_df['bet_date'] = pd.to_datetime(betting_df['bet_date'])
        
        sessions_df = pd.read_csv(f"{data_path}/sessions.csv")
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
        
        campaigns_df = pd.read_csv(f"{data_path}/campaigns.csv")
        campaigns_df['start_date'] = pd.to_datetime(campaigns_df['start_date'])
        campaigns_df['end_date'] = pd.to_datetime(campaigns_df['end_date'])
        
        customer_campaigns_df = pd.read_csv(f"{data_path}/customer_campaigns.csv")
        customer_campaigns_df['sent_date'] = pd.to_datetime(customer_campaigns_df['sent_date'])
        
        return {
            'customers': customers_df,
            'transactions': transactions_df,
            'betting': betting_df,
            'sessions': sessions_df,
            'campaigns': campaigns_df,
            'customer_campaigns': customer_campaigns_df
        }
    except FileNotFoundError:
        return None

def calculate_kpis(data):
    """Calculate key marketing KPIs"""
    customers_df = data['customers']
    transactions_df = data['transactions']
    betting_df = data['betting']
    
    # Basic metrics
    total_customers = len(customers_df)
    converted_customers = len(transactions_df[transactions_df['is_first_deposit'] == True]['customer_id'].unique())
    total_deposits = transactions_df[transactions_df['transaction_type'] == 'deposit']['amount'].sum()
    total_wagered = betting_df['bet_amount'].sum()
    gross_gaming_revenue = (betting_df['bet_amount'] - betting_df['payout']).sum()
    
    # Conversion rate
    conversion_rate = converted_customers / total_customers if total_customers > 0 else 0
    
    # Average customer metrics
    avg_customer_ltv = gross_gaming_revenue / converted_customers if converted_customers > 0 else 0
    avg_first_deposit = transactions_df[transactions_df['is_first_deposit'] == True]['amount'].mean()
    
    # Channel performance
    channel_performance = customers_df.groupby('acquisition_channel').agg({
        'customer_id': 'count',
        'acquisition_cost': 'sum'
    }).reset_index()
    
    # Calculate conversion by channel
    converted_by_channel = transactions_df[transactions_df['is_first_deposit'] == True].merge(
        customers_df[['customer_id', 'acquisition_channel']], on='customer_id'
    ).groupby('acquisition_channel')['customer_id'].count().reset_index()
    converted_by_channel.columns = ['acquisition_channel', 'conversions']
    
    channel_performance = channel_performance.merge(converted_by_channel, on='acquisition_channel', how='left')
    channel_performance['conversions'] = channel_performance['conversions'].fillna(0)
    channel_performance['conversion_rate'] = channel_performance['conversions'] / channel_performance['customer_id']
    channel_performance['cpa'] = channel_performance['acquisition_cost'] / channel_performance['conversions']
    channel_performance['cpa'] = channel_performance['cpa'].replace([np.inf], 0)
    
    return {
        'total_customers': total_customers,
        'converted_customers': converted_customers,
        'conversion_rate': conversion_rate,
        'total_deposits': total_deposits,
        'total_wagered': total_wagered,
        'gross_gaming_revenue': gross_gaming_revenue,
        'avg_customer_ltv': avg_customer_ltv,
        'avg_first_deposit': avg_first_deposit,
        'channel_performance': channel_performance
    }

def create_daily_metrics(data):
    """Create daily metrics for trend analysis"""
    betting_df = data['betting']
    transactions_df = data['transactions']
    
    # Daily betting metrics
    daily_betting = betting_df.groupby(betting_df['bet_date'].dt.date).agg({
        'customer_id': 'nunique',
        'bet_id': 'count',
        'bet_amount': 'sum',
        'payout': 'sum'
    }).reset_index()
    daily_betting.columns = ['date', 'daily_active_users', 'total_bets', 'total_wagered', 'total_payouts']
    daily_betting['gross_gaming_revenue'] = daily_betting['total_wagered'] - daily_betting['total_payouts']
    
    # Daily transaction metrics
    daily_deposits = transactions_df[transactions_df['transaction_type'] == 'deposit'].groupby(
        transactions_df['transaction_date'].dt.date
    ).agg({
        'customer_id': 'nunique',
        'amount': 'sum'
    }).reset_index()
    daily_deposits.columns = ['date', 'depositing_users', 'total_deposits']
    
    # Merge
    daily_metrics = daily_betting.merge(daily_deposits, on='date', how='left')
    daily_metrics = daily_metrics.fillna(0)
    
    return daily_metrics

def main():
    # Header
    st.markdown('<h1 class="main-header">🎰 Sportsbook Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    if data is None:
        st.error("❌ Data not found! Please run the data generator first:")
        st.code("python python/data_generator.py")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">📊 Dashboard Controls</div>', unsafe_allow_html=True)
    
    # Date range filter
    min_date = data['customers']['signup_date'].min().date()
    max_date = data['customers']['signup_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Channel filter
    channels = ['All'] + list(data['customers']['acquisition_channel'].unique())
    selected_channel = st.sidebar.selectbox("Select Channel", channels)
    
    # Customer segment filter
    segments = ['All'] + list(data['customers']['customer_segment'].unique())
    selected_segment = st.sidebar.selectbox("Select Customer Segment", segments)
    
    # Filter data based on selections
    filtered_customers = data['customers'].copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_customers = filtered_customers[
            (filtered_customers['signup_date'].dt.date >= start_date) &
            (filtered_customers['signup_date'].dt.date <= end_date)
        ]
    
    if selected_channel != 'All':
        filtered_customers = filtered_customers[filtered_customers['acquisition_channel'] == selected_channel]
    
    if selected_segment != 'All':
        filtered_customers = filtered_customers[filtered_customers['customer_segment'] == selected_segment]
    
    # Update other datasets based on filtered customers
    customer_ids = filtered_customers['customer_id'].tolist()
    filtered_data = {
        'customers': filtered_customers,
        'transactions': data['transactions'][data['transactions']['customer_id'].isin(customer_ids)],
        'betting': data['betting'][data['betting']['customer_id'].isin(customer_ids)],
        'sessions': data['sessions'][data['sessions']['customer_id'].isin(customer_ids)],
        'campaigns': data['campaigns'],
        'customer_campaigns': data['customer_campaigns'][data['customer_campaigns']['customer_id'].isin(customer_ids)]
    }
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_data)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🔄 Cohort Analysis", "🧪 A/B Testing", 
        "🎯 Attribution", "🤖 ML Insights"
    ])
    
    with tab1:
        st.header("Marketing Overview")
        
        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Customers",
                f"{kpis['total_customers']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Conversion Rate",
                f"{kpis['conversion_rate']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Total Deposits",
                f"${kpis['total_deposits']:,.0f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Gross Gaming Revenue",
                f"${kpis['gross_gaming_revenue']:,.0f}",
                delta=None
            )
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Channel Performance")
            
            # Channel performance chart
            fig = px.bar(
                kpis['channel_performance'],
                x='acquisition_channel',
                y='customer_id',
                title="Customers by Acquisition Channel",
                labels={'customer_id': 'Number of Customers', 'acquisition_channel': 'Channel'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Conversion Rates by Channel")
            
            fig = px.bar(
                kpis['channel_performance'],
                x='acquisition_channel',
                y='conversion_rate',
                title="Conversion Rate by Channel",
                labels={'conversion_rate': 'Conversion Rate', 'acquisition_channel': 'Channel'}
            )
            fig.update_layout(height=400, yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily trends
        st.subheader("Daily Performance Trends")
        
        daily_metrics = create_daily_metrics(filtered_data)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily Active Users', 'Total Wagered', 'Gross Gaming Revenue', 'Total Deposits'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['daily_active_users'], name='Daily Active Users'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['total_wagered'], name='Total Wagered'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['gross_gaming_revenue'], name='GGR'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['total_deposits'], name='Total Deposits'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Cohort Analysis")
        
        try:
            # Initialize cohort analyzer
            analyzer = CohortAnalyzer()
            analyzer.customers_df = filtered_data['customers']
            analyzer.betting_df = filtered_data['betting']
            analyzer.transactions_df = filtered_data['transactions']
            
            # Create retention cohorts
            retention_matrix = analyzer.create_retention_cohorts('monthly')
            
            st.subheader("Monthly Retention Cohorts")
            
            # Interactive heatmap
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
                title="Customer Retention Heatmap",
                xaxis_title="Period Number (0 = Signup Period)",
                yaxis_title="Cohort Period",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.subheader("Retention Insights")
            
            if not retention_matrix.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    period_1_retention = retention_matrix[1].mean() if 1 in retention_matrix.columns else 0
                    st.metric("Average 1-Month Retention", f"{period_1_retention:.1%}")
                
                with col2:
                    period_3_retention = retention_matrix[3].mean() if 3 in retention_matrix.columns else 0
                    st.metric("Average 3-Month Retention", f"{period_3_retention:.1%}")
                
                with col3:
                    period_6_retention = retention_matrix[6].mean() if 6 in retention_matrix.columns else 0
                    st.metric("Average 6-Month Retention", f"{period_6_retention:.1%}")
        
        except Exception as e:
            st.error(f"Error in cohort analysis: {e}")
    
    with tab3:
        st.header("A/B Testing Simulation")
        
        st.subheader("Campaign Simulator")
        
        # A/B test configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Control Group**")
            control_size = st.number_input("Control Group Size", min_value=100, max_value=5000, value=1000)
            control_conversion = st.slider("Control Conversion Rate", 0.01, 0.30, 0.08, 0.01)
        
        with col2:
            st.markdown("**Treatment Group**")
            treatment_size = st.number_input("Treatment Group Size", min_value=100, max_value=5000, value=1000)
            expected_lift = st.slider("Expected Lift", 0.05, 0.50, 0.15, 0.05)
        
        if st.button("Run A/B Test Simulation"):
            with st.spinner("Running simulation..."):
                # Simulate A/B test
                simulator = CampaignSimulator()
                campaign_data = simulator.simulate_welcome_bonus_test(
                    n_control=control_size,
                    n_treatment=treatment_size,
                    control_conversion_rate=control_conversion,
                    treatment_lift=expected_lift
                )
                
                # Analyze results
                ab_analyzer = ABTestAnalyzer()
                results = ab_analyzer.run_campaign_analysis(campaign_data)
                
                # Display results
                if results:
                    st.subheader("A/B Test Results")
                    
                    for metric_name, result in results.items():
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                f"{result.metric_name} - Control",
                                f"{result.control_value:.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                f"{result.metric_name} - Treatment",
                                f"{result.treatment_value:.4f}",
                                delta=f"{result.lift:.4f}"
                            )
                        
                        with col3:
                            significance = "✅ Significant" if result.is_significant else "❌ Not Significant"
                            st.metric(
                                "Statistical Significance",
                                significance,
                                delta=f"p-value: {result.p_value:.4f}"
                            )
                        
                        st.markdown("---")
    
    with tab4:
        st.header("Marketing Attribution Analysis")
        
        try:
            # Initialize attribution analyzer
            attribution_analyzer = MarketingAttributionAnalyzer()
            attribution_analyzer.customers_df = filtered_data['customers']
            attribution_analyzer.campaigns_df = filtered_data['campaigns']
            attribution_analyzer.customer_campaigns_df = filtered_data['customer_campaigns']
            attribution_analyzer.transactions_df = filtered_data['transactions']
            
            # Create customer journey
            journey_df = attribution_analyzer.create_customer_journey()
            
            if not journey_df.empty:
                st.subheader("Attribution Model Comparison")
                
                # Compare attribution models
                comparison_df = attribution_analyzer.compare_attribution_models(journey_df)
                
                # Create visualization
                pivot_data = comparison_df.pivot(index='channel', columns='model', values='attributed_conversions')
                
                fig = px.bar(
                    comparison_df,
                    x='channel',
                    y='attributed_conversions',
                    color='model',
                    title="Attributed Conversions by Model",
                    barmode='group'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Attribution insights
                st.subheader("Attribution Insights")
                
                # Show summary table
                summary_table = comparison_df.pivot_table(
                    index='channel',
                    columns='model',
                    values='attributed_conversions',
                    fill_value=0
                ).round(1)
                
                st.dataframe(summary_table)
            
            else:
                st.warning("No customer journey data available for attribution analysis.")
        
        except Exception as e:
            st.error(f"Error in attribution analysis: {e}")
    
    with tab5:
        st.header("Machine Learning Insights")
        
        # ML model status
        model_path = "models"
        models_exist = any(
            os.path.exists(f"{model_path}/{model}.pkl") 
            for model in ['churn_prediction', 'ltv_forecasting', 'conversion_prediction']
        )
        
        if models_exist:
            st.success("✅ ML Models are available")
            
            try:
                # Initialize ML predictor
                predictor = SportsbookMLPredictor()
                predictor.customers_df = filtered_data['customers']
                predictor.transactions_df = filtered_data['transactions']
                predictor.betting_df = filtered_data['betting']
                predictor.sessions_df = filtered_data['sessions']
                
                # Load models
                predictor.load_models()
                
                # Generate predictions for sample customers
                sample_customers = filtered_data['customers']['customer_id'].head(100).tolist()
                risk_scores = predictor.predict_customer_risk_scores(sample_customers)
                
                if not risk_scores.empty:
                    st.subheader("Customer Risk Scores")
                    
                    # Risk distribution
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'churn_risk' in risk_scores.columns:
                            churn_dist = risk_scores['churn_risk'].value_counts()
                            fig = px.pie(values=churn_dist.values, names=churn_dist.index, title="Churn Risk Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'ltv_segment' in risk_scores.columns:
                            ltv_dist = risk_scores['ltv_segment'].value_counts()
                            fig = px.pie(values=ltv_dist.values, names=ltv_dist.index, title="LTV Segment Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        if 'conversion_potential' in risk_scores.columns:
                            conv_dist = risk_scores['conversion_potential'].value_counts()
                            fig = px.pie(values=conv_dist.values, names=conv_dist.index, title="Conversion Potential Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk scores table
                    st.subheader("Sample Customer Risk Scores")
                    display_cols = ['customer_id', 'churn_probability', 'churn_risk', 
                                  'predicted_ltv', 'ltv_segment', 'conversion_probability', 'customer_segment']
                    available_cols = [col for col in display_cols if col in risk_scores.columns]
                    
                    if available_cols:
                        st.dataframe(
                            risk_scores[available_cols].head(20).style.format({
                                'churn_probability': '{:.2%}',
                                'predicted_ltv': '${:.2f}',
                                'conversion_probability': '{:.2%}'
                            }),
                            use_container_width=True
                        )
                
                else:
                    st.warning("No risk scores generated. Please check the data.")
            
            except Exception as e:
                st.error(f"Error in ML predictions: {e}")
        
        else:
            st.warning("🚧 ML Models not found. Please train the models first:")
            #st.code("python python/ml_predictive_models.py")
            
            # Disabled button with coming soon message
            st.button("🔄 Train Models Now - Coming Soon", disabled=True, help="Model training feature is being finalized. Please use the command line for now.")
            
            st.info("💡 **Still working on this feature to generate ML models.**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Sportsbook Marketing Analytics Dashboard** | "
        "Built with Streamlit | "
        f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()