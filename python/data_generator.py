"""
Sportsbook Marketing Data Generator

This module generates realistic synthetic data for a sportsbook platform,
including customer demographics, betting behavior, marketing campaigns,
and transaction history.

Author: Sportsbook Marketing Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import json
from typing import Dict, List, Tuple
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

class SportsbookDataGenerator:
    def __init__(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.days_range = (self.end_date - self.start_date).days
        
        # Marketing channels with different performance characteristics
        self.channels = {
            'paid_search': {'cost_per_click': 3.5, 'conversion_rate': 0.08, 'weight': 0.35},
            'social_media': {'cost_per_click': 2.1, 'conversion_rate': 0.05, 'weight': 0.20},
            'affiliate': {'cost_per_click': 4.2, 'conversion_rate': 0.12, 'weight': 0.28},
            'organic': {'cost_per_click': 0.0, 'conversion_rate': 0.15, 'weight': 0.17}
        }
        
        # Sport categories for betting
        self.sports = ['football', 'basketball', 'baseball', 'soccer', 'tennis', 'hockey', 'golf', 'mma']
        
        # Customer segments with different behaviors
        self.customer_segments = {
            'casual': {'bet_frequency': 2.5, 'avg_bet_size': 25, 'ltv_multiplier': 1.0, 'weight': 0.65},
            'regular': {'bet_frequency': 8.2, 'avg_bet_size': 75, 'ltv_multiplier': 2.8, 'weight': 0.25},
            'whale': {'bet_frequency': 15.3, 'avg_bet_size': 250, 'ltv_multiplier': 8.5, 'weight': 0.10}
        }

    def generate_customers(self, num_customers: int) -> pd.DataFrame:
        """Generate synthetic customer data"""
        print(f"Generating {num_customers:,} customers...")
        
        customers = []
        
        for i in range(num_customers):
            # Random signup date
            signup_date = self.start_date + timedelta(days=random.randint(0, self.days_range))
            
            # Choose customer segment
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=list(seg['weight'] for seg in self.customer_segments.values())
            )
            
            # Choose acquisition channel
            channel = np.random.choice(
                list(self.channels.keys()),
                p=list(ch['weight'] for ch in self.channels.values())
            )
            
            # Calculate acquisition cost
            if channel == 'organic':
                acquisition_cost = 0
            else:
                base_cost = self.channels[channel]['cost_per_click'] * random.uniform(8, 25)
                acquisition_cost = round(base_cost * random.uniform(0.7, 1.4), 2)
            
            # Customer demographics
            age = np.random.normal(32, 8)
            age = max(18, min(65, int(age)))  # Clamp between 18-65
            
            customer = {
                'customer_id': f"CU_{i+1:06d}",
                'signup_date': signup_date,
                'acquisition_channel': channel,
                'acquisition_cost': acquisition_cost,
                'age': age,
                'gender': random.choice(['M', 'F', 'O']),
                'country': fake.country_code(),
                'state': fake.state_abbr(),
                'city': fake.city(),
                'customer_segment': segment,
                'email_verified': random.choices([True, False], weights=[0.85, 0.15])[0],
                'phone_verified': random.choices([True, False], weights=[0.72, 0.28])[0],
            }
            
            customers.append(customer)
            
            if (i + 1) % 5000 == 0:
                print(f"  Generated {i+1:,} customers")
        
        return pd.DataFrame(customers)

    def generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate deposit and withdrawal transactions"""
        print("Generating transaction data...")
        
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            signup_date = customer['signup_date']
            segment = customer['customer_segment']
            
            # Determine if customer makes initial deposit (conversion)
            channel = customer['acquisition_channel']
            conversion_rate = self.channels[channel]['conversion_rate']
            
            # Adjust conversion rate by segment
            if segment == 'whale':
                conversion_rate *= 1.4
            elif segment == 'regular':
                conversion_rate *= 1.2
            
            if random.random() > conversion_rate:
                continue  # Customer doesn't convert
            
            # First deposit (usually within 7 days of signup)
            first_deposit_date = signup_date + timedelta(days=random.randint(0, 7))
            
            # First deposit amount varies by segment
            if segment == 'casual':
                first_deposit = round(random.uniform(20, 100), 2)
            elif segment == 'regular':
                first_deposit = round(random.uniform(100, 500), 2)
            else:  # whale
                first_deposit = round(random.uniform(500, 2000), 2)
            
            transactions.append({
                'transaction_id': f"TXN_{transaction_id:08d}",
                'customer_id': customer_id,
                'transaction_type': 'deposit',
                'amount': first_deposit,
                'transaction_date': first_deposit_date,
                'payment_method': random.choice(['credit_card', 'debit_card', 'bank_transfer', 'paypal']),
                'is_first_deposit': True
            })
            transaction_id += 1
            
            # Generate subsequent transactions
            current_date = first_deposit_date
            days_since_signup = 0
            
            # Simulate customer lifecycle with decreasing activity over time
            while current_date < self.end_date and days_since_signup < 365:
                days_since_signup = (current_date - signup_date).days
                
                # Probability of transaction decreases over time (churn simulation)
                activity_decay = max(0.1, 1 - (days_since_signup / 730))  # 2-year decay
                
                # Segment-based activity levels
                if segment == 'casual':
                    transaction_prob = 0.15 * activity_decay
                elif segment == 'regular':
                    transaction_prob = 0.35 * activity_decay
                else:  # whale
                    transaction_prob = 0.55 * activity_decay
                
                if random.random() < transaction_prob:
                    # Determine transaction type (deposits more frequent than withdrawals)
                    trans_type = random.choices(['deposit', 'withdrawal'], weights=[0.7, 0.3])[0]
                    
                    # Amount varies by segment and type
                    if trans_type == 'deposit':
                        if segment == 'casual':
                            amount = round(random.uniform(25, 150), 2)
                        elif segment == 'regular':
                            amount = round(random.uniform(100, 400), 2)
                        else:  # whale
                            amount = round(random.uniform(300, 1500), 2)
                    else:  # withdrawal
                        # Withdrawals typically smaller and less frequent
                        if segment == 'casual':
                            amount = round(random.uniform(50, 200), 2)
                        elif segment == 'regular':
                            amount = round(random.uniform(150, 600), 2)
                        else:  # whale
                            amount = round(random.uniform(400, 2000), 2)
                    
                    transactions.append({
                        'transaction_id': f"TXN_{transaction_id:08d}",
                        'customer_id': customer_id,
                        'transaction_type': trans_type,
                        'amount': amount,
                        'transaction_date': current_date,
                        'payment_method': random.choice(['credit_card', 'debit_card', 'bank_transfer', 'paypal']),
                        'is_first_deposit': False
                    })
                    transaction_id += 1
                
                # Move to next potential transaction date
                current_date += timedelta(days=random.randint(1, 14))
        
        print(f"Generated {len(transactions):,} transactions")
        return pd.DataFrame(transactions)

    def generate_betting_activity(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate betting activity data"""
        print("Generating betting activity...")
        
        # Only customers who made deposits can place bets
        depositing_customers = transactions_df[
            transactions_df['transaction_type'] == 'deposit'
        ]['customer_id'].unique()
        
        betting_activity = []
        bet_id = 1
        
        for customer_id in depositing_customers:
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            segment = customer['customer_segment']
            signup_date = customer['signup_date']
            
            # Get customer's deposit dates to simulate realistic betting patterns
            customer_deposits = transactions_df[
                (transactions_df['customer_id'] == customer_id) & 
                (transactions_df['transaction_type'] == 'deposit')
            ].sort_values('transaction_date')
            
            current_date = customer_deposits.iloc[0]['transaction_date']
            days_active = 0
            
            while current_date < self.end_date and days_active < 365:
                days_since_signup = (current_date - signup_date).days
                activity_decay = max(0.1, 1 - (days_since_signup / 365))
                
                # Betting frequency by segment
                if segment == 'casual':
                    bet_prob = 0.25 * activity_decay
                    bets_per_session = random.randint(1, 3)
                elif segment == 'regular':
                    bet_prob = 0.45 * activity_decay
                    bets_per_session = random.randint(2, 6)
                else:  # whale
                    bet_prob = 0.65 * activity_decay
                    bets_per_session = random.randint(3, 12)
                
                if random.random() < bet_prob:
                    session_id = f"SES_{bet_id:08d}"
                    session_start = current_date + timedelta(
                        hours=random.randint(9, 23),
                        minutes=random.randint(0, 59)
                    )
                    
                    # Generate bets for this session
                    for bet_num in range(bets_per_session):
                        sport = random.choice(self.sports)
                        
                        # Bet amount varies by segment
                        base_bet = self.customer_segments[segment]['avg_bet_size']
                        bet_amount = round(random.uniform(base_bet * 0.3, base_bet * 2.5), 2)
                        
                        # Outcome (house edge simulation - sportsbook wins ~52% of the time)
                        win_probability = 0.48  # Customer win rate
                        bet_won = random.random() < win_probability
                        
                        if bet_won:
                            # Typical odds around 1.8 to 2.2
                            odds = random.uniform(1.8, 2.2)
                            payout = round(bet_amount * odds, 2)
                            net_result = payout - bet_amount
                        else:
                            payout = 0
                            net_result = -bet_amount
                        
                        betting_activity.append({
                            'bet_id': f"BET_{bet_id:010d}",
                            'customer_id': customer_id,
                            'session_id': session_id,
                            'sport': sport,
                            'bet_amount': bet_amount,
                            'bet_date': session_start + timedelta(minutes=bet_num * random.randint(5, 20)),
                            'bet_won': bet_won,
                            'payout': payout,
                            'net_result': net_result,
                            'odds': odds if bet_won else random.uniform(1.5, 3.0)
                        })
                        bet_id += 1
                
                current_date += timedelta(days=random.randint(1, 7))
                days_active += 1
        
        print(f"Generated {len(betting_activity):,} bets")
        return pd.DataFrame(betting_activity)

    def generate_marketing_campaigns(self, customers_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate marketing campaigns and customer responses"""
        print("Generating marketing campaigns...")
        
        campaigns = []
        customer_campaigns = []
        
        # Define campaign types
        campaign_types = [
            {
                'name': 'Welcome Bonus',
                'type': 'signup_bonus',
                'description': 'Deposit $50, Get $50 Bonus',
                'cost_per_customer': 15.0,
                'bonus_amount': 50,
                'min_deposit': 50
            },
            {
                'name': 'Reload Bonus',
                'type': 'retention',
                'description': '25% Reload Bonus up to $100',
                'cost_per_customer': 8.5,
                'bonus_amount': 25,  # percentage
                'min_deposit': 100
            },
            {
                'name': 'Free Bet Friday',
                'type': 'engagement',
                'description': '$10 Free Bet Every Friday',
                'cost_per_customer': 10.0,
                'bonus_amount': 10,
                'min_deposit': 0
            },
            {
                'name': 'High Roller Bonus',
                'type': 'vip',
                'description': 'Deposit $500, Get $200 Bonus',
                'cost_per_customer': 45.0,
                'bonus_amount': 200,
                'min_deposit': 500
            }
        ]
        
        campaign_id = 1
        
        # Generate monthly campaigns
        current_month = self.start_date.replace(day=1)
        
        while current_month <= self.end_date:
            # 2-3 campaigns per month
            num_campaigns = random.randint(2, 3)
            
            for _ in range(num_campaigns):
                campaign_type = random.choice(campaign_types)
                
                start_date = current_month + timedelta(days=random.randint(1, 28))
                end_date = start_date + timedelta(days=random.randint(7, 21))
                
                # Target different customer segments
                if campaign_type['type'] == 'vip':
                    target_segments = ['whale']
                elif campaign_type['type'] == 'retention':
                    target_segments = ['regular', 'whale']
                else:
                    target_segments = ['casual', 'regular', 'whale']
                
                campaign = {
                    'campaign_id': f"CAMP_{campaign_id:05d}",
                    'campaign_name': f"{campaign_type['name']} - {start_date.strftime('%B %Y')}",
                    'campaign_type': campaign_type['type'],
                    'start_date': start_date,
                    'end_date': end_date,
                    'description': campaign_type['description'],
                    'cost_per_customer': campaign_type['cost_per_customer'],
                    'bonus_amount': campaign_type['bonus_amount'],
                    'min_deposit': campaign_type['min_deposit'],
                    'target_segments': ','.join(target_segments)
                }
                campaigns.append(campaign)
                
                # Assign customers to campaign (A/B testing simulation)
                eligible_customers = customers_df[
                    (customers_df['customer_segment'].isin(target_segments)) &
                    (customers_df['signup_date'] <= start_date)
                ]
                
                # Random sample for campaign (50-80% of eligible customers)
                sample_rate = random.uniform(0.5, 0.8)
                campaign_customers = eligible_customers.sample(frac=sample_rate)
                
                for _, customer in campaign_customers.iterrows():
                    # Response rate varies by campaign type and segment
                    base_response = {
                        'signup_bonus': 0.35,
                        'retention': 0.18,
                        'engagement': 0.12,
                        'vip': 0.45
                    }[campaign_type['type']]
                    
                    # Adjust by segment
                    if customer['customer_segment'] == 'whale':
                        response_rate = base_response * 1.3
                    elif customer['customer_segment'] == 'regular':
                        response_rate = base_response * 1.1
                    else:
                        response_rate = base_response
                    
                    responded = random.random() < response_rate
                    
                    customer_campaigns.append({
                        'customer_id': customer['customer_id'],
                        'campaign_id': campaign['campaign_id'],
                        'sent_date': start_date + timedelta(days=random.randint(0, 3)),
                        'responded': responded,
                        'response_date': start_date + timedelta(days=random.randint(1, 10)) if responded else None,
                        'bonus_claimed': responded and random.random() < 0.8,  # 80% claim rate
                        'campaign_cost': campaign_type['cost_per_customer']
                    })
                
                campaign_id += 1
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        print(f"Generated {len(campaigns)} campaigns with {len(customer_campaigns):,} customer interactions")
        return pd.DataFrame(campaigns), pd.DataFrame(customer_campaigns)

    def generate_sessions(self, customers_df: pd.DataFrame, betting_df: pd.DataFrame) -> pd.DataFrame:
        """Generate user session data"""
        print("Generating session data...")
        
        sessions = []
        session_id = 1
        
        # Get customers who have betting activity
        active_customers = betting_df['customer_id'].unique()
        
        for customer_id in active_customers:
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            customer_bets = betting_df[betting_df['customer_id'] == customer_id].sort_values('bet_date')
            
            # Group bets into sessions (bets within 2 hours = same session)
            current_session = None
            session_bets = []
            
            for _, bet in customer_bets.iterrows():
                bet_time = bet['bet_date']
                
                if current_session is None or (bet_time - current_session['start_time']).total_seconds() > 7200:
                    # New session
                    if current_session is not None:
                        # Finalize previous session
                        current_session['end_time'] = session_bets[-1]['bet_date']
                        current_session['duration_minutes'] = int(
                            (current_session['end_time'] - current_session['start_time']).total_seconds() / 60
                        )
                        current_session['num_bets'] = len(session_bets)
                        current_session['total_wagered'] = sum(b['bet_amount'] for b in session_bets)
                        current_session['net_result'] = sum(b['net_result'] for b in session_bets)
                        
                        sessions.append(current_session)
                    
                    # Start new session
                    current_session = {
                        'session_id': f"SES_{session_id:08d}",
                        'customer_id': customer_id,
                        'start_time': bet_time,
                        'device_type': random.choices(['mobile', 'desktop', 'tablet'], weights=[0.65, 0.25, 0.10])[0],
                        'browser': random.choices(['Chrome', 'Safari', 'Firefox', 'Edge'], weights=[0.55, 0.25, 0.15, 0.05])[0],
                        'country': customer['country']
                    }
                    session_bets = []
                    session_id += 1
                
                session_bets.append(bet.to_dict())
            
            # Don't forget the last session
            if current_session is not None and session_bets:
                current_session['end_time'] = session_bets[-1]['bet_date']
                current_session['duration_minutes'] = int(
                    (current_session['end_time'] - current_session['start_time']).total_seconds() / 60
                )
                current_session['num_bets'] = len(session_bets)
                current_session['total_wagered'] = sum(b['bet_amount'] for b in session_bets)
                current_session['net_result'] = sum(b['net_result'] for b in session_bets)
                
                sessions.append(current_session)
        
        print(f"Generated {len(sessions):,} sessions")
        return pd.DataFrame(sessions)

    def save_data(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/raw"):
        """Save all datasets to CSV files"""
        print(f"Saving datasets to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, df in datasets.items():
            filename = f"{output_dir}/{dataset_name}.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved {dataset_name}: {len(df):,} records → {filename}")
        
        # Save data dictionary
        data_dict = {
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'date_range': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            'total_customers': len(datasets['customers']),
            'total_transactions': len(datasets['transactions']),
            'total_bets': len(datasets['betting_activity']),
            'total_sessions': len(datasets['sessions']),
            'total_campaigns': len(datasets['campaigns']),
            'channels': list(self.channels.keys()),
            'sports': self.sports,
            'customer_segments': list(self.customer_segments.keys())
        }
        
        with open(f"{output_dir}/data_dictionary.json", "w") as f:
            json.dump(data_dict, f, indent=2, default=str)
        
        print(f"  Saved data dictionary → {output_dir}/data_dictionary.json")

def main():
    """Generate complete sportsbook dataset"""
    print("🎰 Sportsbook Marketing Data Generator")
    print("=" * 50)
    
    # Configuration
    NUM_CUSTOMERS = 10000
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"
    OUTPUT_DIR = "data/raw"
    
    # Initialize generator
    generator = SportsbookDataGenerator(START_DATE, END_DATE)
    
    # Generate datasets
    print(f"\nGenerating synthetic data for {NUM_CUSTOMERS:,} customers...")
    
    # 1. Generate customers
    customers_df = generator.generate_customers(NUM_CUSTOMERS)
    
    # 2. Generate transactions
    transactions_df = generator.generate_transactions(customers_df)
    
    # 3. Generate betting activity
    betting_df = generator.generate_betting_activity(customers_df, transactions_df)
    
    # 4. Generate marketing campaigns
    campaigns_df, customer_campaigns_df = generator.generate_marketing_campaigns(customers_df)
    
    # 5. Generate sessions
    sessions_df = generator.generate_sessions(customers_df, betting_df)
    
    # Combine all datasets
    datasets = {
        'customers': customers_df,
        'transactions': transactions_df,
        'betting_activity': betting_df,
        'campaigns': campaigns_df,
        'customer_campaigns': customer_campaigns_df,
        'sessions': sessions_df
    }
    
    # Save to files
    generator.save_data(datasets, OUTPUT_DIR)
    
    print("\n📊 Data Generation Summary:")
    print(f"  Customers: {len(customers_df):,}")
    print(f"  Transactions: {len(transactions_df):,}")
    print(f"  Bets: {len(betting_df):,}")
    print(f"  Sessions: {len(sessions_df):,}")
    print(f"  Campaigns: {len(campaigns_df):,}")
    print(f"  Campaign Interactions: {len(customer_campaigns_df):,}")
    
    # Quick stats
    conversion_rate = len(transactions_df[transactions_df['is_first_deposit']]) / len(customers_df)
    avg_bet_size = betting_df['bet_amount'].mean()
    
    print(f"\n🔍 Key Metrics:")
    print(f"  Overall Conversion Rate: {conversion_rate:.1%}")
    print(f"  Average Bet Size: ${avg_bet_size:.2f}")
    print(f"  Total Volume Wagered: ${betting_df['bet_amount'].sum():,.2f}")
    
    print("\n✅ Data generation complete! Files saved to 'data/raw/' directory.")
    print("\nNext steps:")
    print("  1. Run SQL analytics: python python/sql_analytics.py")
    print("  2. Launch dashboard: streamlit run dashboards/streamlit_dashboard.py")

if __name__ == "__main__":
    main()