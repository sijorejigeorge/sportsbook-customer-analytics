"""
Marketing Attribution Models for Sportsbook Analytics

This module implements various attribution models to understand the
customer journey and assign credit to different marketing touchpoints.
Includes last-click, first-touch, linear, and time-decay attribution.

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

class MarketingAttributionAnalyzer:
    """
    Comprehensive marketing attribution analysis including multiple attribution models
    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the attribution analyzer
        
        Args:
            data_path: Path to the raw data files
        """
        self.data_path = data_path
        self.customers_df = None
        self.transactions_df = None
        self.campaigns_df = None
        self.customer_campaigns_df = None
        
        # Attribution models
        self.attribution_models = {
            'last_click': self._last_click_attribution,
            'first_touch': self._first_touch_attribution,
            'linear': self._linear_attribution,
            'time_decay': self._time_decay_attribution,
            'position_based': self._position_based_attribution
        }
    
    def load_data(self):
        """Load customer, campaign, and transaction data"""
        try:
            print("Loading data for attribution analysis...")
            
            self.customers_df = pd.read_csv(f"{self.data_path}/customers.csv")
            self.customers_df['signup_date'] = pd.to_datetime(self.customers_df['signup_date'])
            
            self.transactions_df = pd.read_csv(f"{self.data_path}/transactions.csv")
            self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
            
            self.campaigns_df = pd.read_csv(f"{self.data_path}/campaigns.csv")
            self.campaigns_df['start_date'] = pd.to_datetime(self.campaigns_df['start_date'])
            self.campaigns_df['end_date'] = pd.to_datetime(self.campaigns_df['end_date'])
            
            self.customer_campaigns_df = pd.read_csv(f"{self.data_path}/customer_campaigns.csv")
            self.customer_campaigns_df['sent_date'] = pd.to_datetime(self.customer_campaigns_df['sent_date'])
            self.customer_campaigns_df['response_date'] = pd.to_datetime(self.customer_campaigns_df['response_date'])
            
            print(f"  Loaded {len(self.customers_df):,} customers")
            print(f"  Loaded {len(self.campaigns_df):,} campaigns")
            print(f"  Loaded {len(self.customer_campaigns_df):,} campaign interactions")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run the data generator first: python python/data_generator.py")
    
    def create_customer_journey(self) -> pd.DataFrame:
        """
        Create customer journey data with all touchpoints
        
        Returns:
            DataFrame with customer journey information
        """
        if self.customers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Creating customer journey data...")
        
        journey_data = []
        
        for _, customer in self.customers_df.iterrows():
            customer_id = customer['customer_id']
            signup_date = customer['signup_date']
            acquisition_channel = customer['acquisition_channel']
            
            # Get first deposit information
            first_deposits = self.transactions_df[
                (self.transactions_df['customer_id'] == customer_id) &
                (self.transactions_df['is_first_deposit'] == True)
            ]
            
            first_deposit_date = first_deposits['transaction_date'].min() if len(first_deposits) > 0 else None
            first_deposit_amount = first_deposits['amount'].sum() if len(first_deposits) > 0 else 0
            converted = len(first_deposits) > 0
            
            # Get all campaign interactions for this customer
            customer_interactions = self.customer_campaigns_df[
                self.customer_campaigns_df['customer_id'] == customer_id
            ].copy()
            
            # Create touchpoints list
            touchpoints = []
            
            # 1. Acquisition touchpoint (always present)
            touchpoints.append({
                'touchpoint_type': 'acquisition',
                'channel': acquisition_channel,
                'date': signup_date,
                'campaign_id': None,
                'campaign_type': None,
                'responded': True,  # Acquisition always results in signup
                'cost': customer['acquisition_cost']
            })
            
            # 2. Campaign touchpoints
            for _, interaction in customer_interactions.iterrows():
                campaign_info = self.campaigns_df[
                    self.campaigns_df['campaign_id'] == interaction['campaign_id']
                ].iloc[0]
                
                touchpoints.append({
                    'touchpoint_type': 'campaign',
                    'channel': campaign_info['campaign_type'],
                    'date': interaction['sent_date'],
                    'campaign_id': interaction['campaign_id'],
                    'campaign_type': campaign_info['campaign_type'],
                    'responded': interaction['responded'],
                    'cost': interaction['campaign_cost']
                })
            
            # Sort touchpoints by date
            touchpoints_df = pd.DataFrame(touchpoints).sort_values('date')
            
            # Filter touchpoints that happened before first deposit (conversion window)
            if converted and first_deposit_date is not None:
                pre_conversion_touchpoints = touchpoints_df[
                    touchpoints_df['date'] <= first_deposit_date
                ]
            else:
                # For non-converted customers, consider all touchpoints within 90 days
                cutoff_date = signup_date + timedelta(days=90)
                pre_conversion_touchpoints = touchpoints_df[
                    touchpoints_df['date'] <= cutoff_date
                ]
            
            journey_data.append({
                'customer_id': customer_id,
                'acquisition_channel': acquisition_channel,
                'signup_date': signup_date,
                'first_deposit_date': first_deposit_date,
                'first_deposit_amount': first_deposit_amount,
                'converted': converted,
                'touchpoints': pre_conversion_touchpoints.to_dict('records'),
                'num_touchpoints': len(pre_conversion_touchpoints),
                'total_touchpoint_cost': pre_conversion_touchpoints['cost'].sum()
            })
        
        return pd.DataFrame(journey_data)
    
    def _last_click_attribution(self, touchpoints: List[Dict]) -> Dict[str, float]:
        """
        Last-click attribution: 100% credit to the last touchpoint before conversion
        
        Args:
            touchpoints: List of touchpoint dictionaries
            
        Returns:
            Dictionary with channel attribution weights
        """
        if not touchpoints:
            return {}
        
        last_touchpoint = touchpoints[-1]
        return {last_touchpoint['channel']: 1.0}
    
    def _first_touch_attribution(self, touchpoints: List[Dict]) -> Dict[str, float]:
        """
        First-touch attribution: 100% credit to the first touchpoint
        
        Args:
            touchpoints: List of touchpoint dictionaries
            
        Returns:
            Dictionary with channel attribution weights
        """
        if not touchpoints:
            return {}
        
        first_touchpoint = touchpoints[0]
        return {first_touchpoint['channel']: 1.0}
    
    def _linear_attribution(self, touchpoints: List[Dict]) -> Dict[str, float]:
        """
        Linear attribution: Equal credit to all touchpoints
        
        Args:
            touchpoints: List of touchpoint dictionaries
            
        Returns:
            Dictionary with channel attribution weights
        """
        if not touchpoints:
            return {}
        
        attribution = {}
        weight_per_touchpoint = 1.0 / len(touchpoints)
        
        for touchpoint in touchpoints:
            channel = touchpoint['channel']
            attribution[channel] = attribution.get(channel, 0) + weight_per_touchpoint
        
        return attribution
    
    def _time_decay_attribution(self, touchpoints: List[Dict], half_life_days: int = 7) -> Dict[str, float]:
        """
        Time-decay attribution: More recent touchpoints get higher weight
        
        Args:
            touchpoints: List of touchpoint dictionaries
            half_life_days: Number of days for 50% weight decay
            
        Returns:
            Dictionary with channel attribution weights
        """
        if not touchpoints:
            return {}
        
        if len(touchpoints) == 1:
            return {touchpoints[0]['channel']: 1.0}
        
        # Calculate weights based on recency
        conversion_date = pd.to_datetime(touchpoints[-1]['date'])
        weights = []
        
        for touchpoint in touchpoints:
            touchpoint_date = pd.to_datetime(touchpoint['date'])
            days_before_conversion = (conversion_date - touchpoint_date).days
            
            # Exponential decay formula
            weight = 2 ** (-days_before_conversion / half_life_days)
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Aggregate by channel
        attribution = {}
        for touchpoint, weight in zip(touchpoints, normalized_weights):
            channel = touchpoint['channel']
            attribution[channel] = attribution.get(channel, 0) + weight
        
        return attribution
    
    def _position_based_attribution(self, touchpoints: List[Dict], first_weight: float = 0.4, 
                                   last_weight: float = 0.4) -> Dict[str, float]:
        """
        Position-based attribution: Higher weight to first and last touchpoints
        
        Args:
            touchpoints: List of touchpoint dictionaries
            first_weight: Weight for first touchpoint (default 40%)
            last_weight: Weight for last touchpoint (default 40%)
            
        Returns:
            Dictionary with channel attribution weights
        """
        if not touchpoints:
            return {}
        
        if len(touchpoints) == 1:
            return {touchpoints[0]['channel']: 1.0}
        
        if len(touchpoints) == 2:
            return {
                touchpoints[0]['channel']: first_weight + (1 - first_weight - last_weight) / 2,
                touchpoints[1]['channel']: last_weight + (1 - first_weight - last_weight) / 2
            }
        
        # Weight distribution
        middle_weight = 1.0 - first_weight - last_weight
        middle_weight_per_touchpoint = middle_weight / (len(touchpoints) - 2)
        
        attribution = {}
        
        for i, touchpoint in enumerate(touchpoints):
            channel = touchpoint['channel']
            
            if i == 0:  # First touchpoint
                weight = first_weight
            elif i == len(touchpoints) - 1:  # Last touchpoint
                weight = last_weight
            else:  # Middle touchpoints
                weight = middle_weight_per_touchpoint
            
            attribution[channel] = attribution.get(channel, 0) + weight
        
        return attribution
    
    def calculate_attributed_conversions(self, journey_df: pd.DataFrame, 
                                       model_name: str = 'last_click') -> pd.DataFrame:
        """
        Calculate attributed conversions for a specific attribution model
        
        Args:
            journey_df: Output from create_customer_journey()
            model_name: Name of attribution model to use
            
        Returns:
            DataFrame with attributed conversions by channel
        """
        if model_name not in self.attribution_models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.attribution_models.keys())}")
        
        attribution_model = self.attribution_models[model_name]
        
        channel_attributions = {}
        converted_customers = journey_df[journey_df['converted'] == True]
        
        for _, customer in converted_customers.iterrows():
            touchpoints = customer['touchpoints']
            customer_value = customer['first_deposit_amount']  # Use first deposit as conversion value
            
            # Get attribution weights for this customer
            attribution_weights = attribution_model(touchpoints)
            
            # Distribute conversion value across channels
            for channel, weight in attribution_weights.items():
                if channel not in channel_attributions:
                    channel_attributions[channel] = {
                        'attributed_conversions': 0,
                        'attributed_value': 0,
                        'customers': []
                    }
                
                channel_attributions[channel]['attributed_conversions'] += weight
                channel_attributions[channel]['attributed_value'] += weight * customer_value
                channel_attributions[channel]['customers'].append(customer['customer_id'])
        
        # Convert to DataFrame
        attribution_results = []
        for channel, data in channel_attributions.items():
            attribution_results.append({
                'channel': channel,
                'attributed_conversions': data['attributed_conversions'],
                'attributed_value': data['attributed_value'],
                'unique_customers': len(set(data['customers']))
            })
        
        attribution_df = pd.DataFrame(attribution_results)
        attribution_df['avg_attributed_value'] = (
            attribution_df['attributed_value'] / attribution_df['attributed_conversions']
        )
        
        return attribution_df.sort_values('attributed_conversions', ascending=False)
    
    def compare_attribution_models(self, journey_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare results across all attribution models
        
        Args:
            journey_df: Output from create_customer_journey()
            
        Returns:
            DataFrame comparing all attribution models
        """
        print("Comparing attribution models...")
        
        model_results = []
        
        for model_name in self.attribution_models.keys():
            print(f"  Calculating {model_name} attribution...")
            
            attribution_df = self.calculate_attributed_conversions(journey_df, model_name)
            
            for _, row in attribution_df.iterrows():
                model_results.append({
                    'model': model_name,
                    'channel': row['channel'],
                    'attributed_conversions': row['attributed_conversions'],
                    'attributed_value': row['attributed_value'],
                    'avg_attributed_value': row['avg_attributed_value']
                })
        
        return pd.DataFrame(model_results)
    
    def calculate_marketing_roi_by_attribution(self, journey_df: pd.DataFrame, 
                                             attribution_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROI by channel using attribution results
        
        Args:
            journey_df: Output from create_customer_journey()
            attribution_df: Output from calculate_attributed_conversions()
            
        Returns:
            DataFrame with ROI analysis by channel
        """
        # Calculate total costs by channel
        channel_costs = {}
        
        for _, customer in journey_df.iterrows():
            for touchpoint in customer['touchpoints']:
                channel = touchpoint['channel']
                cost = touchpoint['cost']
                
                channel_costs[channel] = channel_costs.get(channel, 0) + cost
        
        # Combine with attribution results
        roi_analysis = []
        
        for _, row in attribution_df.iterrows():
            channel = row['channel']
            attributed_value = row['attributed_value']
            total_cost = channel_costs.get(channel, 0)
            
            roi = ((attributed_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            roas = (attributed_value / total_cost) if total_cost > 0 else 0
            
            roi_analysis.append({
                'channel': channel,
                'total_cost': total_cost,
                'attributed_value': attributed_value,
                'roi_percent': roi,
                'roas_ratio': roas,
                'attributed_conversions': row['attributed_conversions']
            })
        
        return pd.DataFrame(roi_analysis).sort_values('roi_percent', ascending=False)
    
    def analyze_customer_journey_patterns(self, journey_df: pd.DataFrame) -> Dict:
        """
        Analyze common patterns in customer journeys
        
        Args:
            journey_df: Output from create_customer_journey()
            
        Returns:
            Dictionary with journey pattern insights
        """
        print("Analyzing customer journey patterns...")
        
        insights = {}
        
        # Journey length analysis
        converted_customers = journey_df[journey_df['converted'] == True]
        non_converted = journey_df[journey_df['converted'] == False]
        
        insights['journey_length'] = {
            'avg_touchpoints_converted': converted_customers['num_touchpoints'].mean(),
            'avg_touchpoints_non_converted': non_converted['num_touchpoints'].mean(),
            'median_touchpoints_converted': converted_customers['num_touchpoints'].median(),
            'median_touchpoints_non_converted': non_converted['num_touchpoints'].median()
        }
        
        # Most common journey patterns
        journey_patterns = []
        
        for _, customer in journey_df.iterrows():
            touchpoints = customer['touchpoints']
            pattern = ' → '.join([tp['channel'] for tp in touchpoints])
            journey_patterns.append(pattern)
        
        pattern_counts = pd.Series(journey_patterns).value_counts()
        insights['common_patterns'] = pattern_counts.head(10).to_dict()
        
        # Conversion rate by journey length
        conversion_by_length = journey_df.groupby('num_touchpoints').agg({
            'converted': ['count', 'sum', 'mean']
        }).round(3)
        conversion_by_length.columns = ['total_customers', 'conversions', 'conversion_rate']
        insights['conversion_by_length'] = conversion_by_length.to_dict('index')
        
        # Time to conversion analysis
        converted_df = journey_df[journey_df['converted'] == True].copy()
        converted_df['days_to_conversion'] = (
            converted_df['first_deposit_date'] - converted_df['signup_date']
        ).dt.days
        
        insights['time_to_conversion'] = {
            'avg_days': converted_df['days_to_conversion'].mean(),
            'median_days': converted_df['days_to_conversion'].median(),
            'p25_days': converted_df['days_to_conversion'].quantile(0.25),
            'p75_days': converted_df['days_to_conversion'].quantile(0.75)
        }
        
        return insights
    
    def plot_attribution_comparison(self, comparison_df: pd.DataFrame):
        """
        Plot comparison of attribution models
        
        Args:
            comparison_df: Output from compare_attribution_models()
        """
        # Create pivot table for better visualization
        pivot_conversions = comparison_df.pivot(
            index='channel', 
            columns='model', 
            values='attributed_conversions'
        ).fillna(0)
        
        pivot_value = comparison_df.pivot(
            index='channel',
            columns='model', 
            values='attributed_value'
        ).fillna(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Attributed conversions comparison
        pivot_conversions.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Attributed Conversions by Model')
        axes[0].set_xlabel('Channel')
        axes[0].set_ylabel('Attributed Conversions')
        axes[0].legend(title='Attribution Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Attributed value comparison
        pivot_value.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Attributed Value by Model')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('Attributed Value ($)')
        axes[1].legend(title='Attribution Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_attribution_dashboard(self, comparison_df: pd.DataFrame, 
                                   roi_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive attribution dashboard
        
        Args:
            comparison_df: Output from compare_attribution_models()
            roi_df: Output from calculate_marketing_roi_by_attribution()
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Attributed Conversions by Model',
                'Attributed Value by Model',
                'Channel ROI Analysis',
                'ROAS by Channel'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Attributed conversions by model
        models = comparison_df['model'].unique()
        colors = px.colors.qualitative.Set3[:len(models)]
        
        for i, model in enumerate(models):
            model_data = comparison_df[comparison_df['model'] == model]
            
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=model_data['channel'],
                    y=model_data['attributed_conversions'],
                    marker_color=colors[i],
                    legendgroup=model,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Attributed value by model
        for i, model in enumerate(models):
            model_data = comparison_df[comparison_df['model'] == model]
            
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=model_data['channel'],
                    y=model_data['attributed_value'],
                    marker_color=colors[i],
                    legendgroup=model,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. ROI analysis
        fig.add_trace(
            go.Bar(
                x=roi_df['channel'],
                y=roi_df['roi_percent'],
                name='ROI %',
                marker_color='green',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. ROAS analysis
        fig.add_trace(
            go.Bar(
                x=roi_df['channel'],
                y=roi_df['roas_ratio'],
                name='ROAS',
                marker_color='blue',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Marketing Attribution Analysis Dashboard",
            showlegend=True
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Attributed Conversions", row=1, col=1)
        fig.update_yaxes(title_text="Attributed Value ($)", row=1, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
        fig.update_yaxes(title_text="ROAS Ratio", row=2, col=2)
        
        return fig

def main():
    """Example usage of marketing attribution analysis"""
    print("📊 Marketing Attribution Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = MarketingAttributionAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    if analyzer.customers_df is None:
        print("❌ Data not available. Please run data generator first:")
        print("   python python/data_generator.py")
        return
    
    print("\n🔄 Creating Customer Journeys...")
    
    # Create customer journey data
    journey_df = analyzer.create_customer_journey()
    
    print(f"   Analyzed {len(journey_df):,} customer journeys")
    print(f"   Converted customers: {journey_df['converted'].sum():,}")
    print(f"   Conversion rate: {journey_df['converted'].mean():.1%}")
    
    print("\n📈 Running Attribution Analysis...")
    
    # Compare all attribution models
    comparison_df = analyzer.compare_attribution_models(journey_df)
    
    # Calculate ROI using last-click attribution as example
    last_click_attribution = analyzer.calculate_attributed_conversions(journey_df, 'last_click')
    roi_analysis = analyzer.calculate_marketing_roi_by_attribution(journey_df, last_click_attribution)
    
    print("\n🎯 Attribution Model Comparison (Conversions):")
    pivot_conversions = comparison_df.pivot(index='channel', columns='model', values='attributed_conversions')
    print(pivot_conversions.round(1).to_string())
    
    print("\n💰 ROI Analysis by Channel (Last-Click Attribution):")
    roi_display = roi_analysis[['channel', 'total_cost', 'attributed_value', 'roi_percent', 'roas_ratio']]
    print(roi_display.round(2).to_string(index=False))
    
    # Journey pattern analysis
    print("\n🗺️ Customer Journey Insights:")
    insights = analyzer.analyze_customer_journey_patterns(journey_df)
    
    print(f"   Average touchpoints (converted): {insights['journey_length']['avg_touchpoints_converted']:.1f}")
    print(f"   Average touchpoints (non-converted): {insights['journey_length']['avg_touchpoints_non_converted']:.1f}")
    print(f"   Average days to conversion: {insights['time_to_conversion']['avg_days']:.1f}")
    print(f"   Median days to conversion: {insights['time_to_conversion']['median_days']:.1f}")
    
    print("\n📊 Most Common Journey Patterns:")
    for pattern, count in list(insights['common_patterns'].items())[:5]:
        print(f"   {pattern}: {count} customers")
    
    print("\n📈 Generating Visualizations...")
    
    # Create visualizations
    analyzer.plot_attribution_comparison(comparison_df)
    
    # Interactive dashboard
    dashboard = analyzer.create_attribution_dashboard(comparison_df, roi_analysis)
    
    print("\n🔍 Key Findings:")
    
    # Find best performing channel by different attribution models
    model_winners = {}
    for model in comparison_df['model'].unique():
        model_data = comparison_df[comparison_df['model'] == model]
        winner = model_data.loc[model_data['attributed_conversions'].idxmax()]
        model_winners[model] = winner['channel']
    
    print("   Top channel by attribution model:")
    for model, channel in model_winners.items():
        print(f"     {model}: {channel}")
    
    # ROI insights
    best_roi_channel = roi_analysis.loc[roi_analysis['roi_percent'].idxmax()]
    print(f"\n   Highest ROI channel: {best_roi_channel['channel']} ({best_roi_channel['roi_percent']:.1f}%)")
    print(f"   Best ROAS channel: {roi_analysis.loc[roi_analysis['roas_ratio'].idxmax()]['channel']}")
    
    # Attribution model insights
    total_conversions_by_model = comparison_df.groupby('model')['attributed_conversions'].sum()
    print(f"\n   Total attributed conversions by model:")
    for model, total in total_conversions_by_model.items():
        print(f"     {model}: {total:.1f}")
    
    print("\n✅ Marketing attribution analysis complete!")
    print("\nActionable insights:")
    print("  1. Compare attribution models to understand channel contribution")
    print("  2. Optimize budget allocation based on multi-touch attribution")
    print("  3. Focus on high-ROI channels identified in the analysis")
    print("  4. Analyze journey patterns to improve customer experience")

if __name__ == "__main__":
    main()