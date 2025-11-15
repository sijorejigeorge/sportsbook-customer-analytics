"""
Predictive Machine Learning Models for Sportsbook Analytics

This module implements ML models for customer churn prediction, 
LTV forecasting, and conversion probability modeling using 
various algorithms and feature engineering techniques.

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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import xgboost as xgb
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import os

warnings.filterwarnings('ignore')

class SportsbookMLPredictor:
    """
    Machine Learning prediction system for sportsbook customer analytics
    """
    
    def __init__(self, data_path: str = "data/raw", model_path: str = "models"):
        """
        Initialize the ML predictor
        
        Args:
            data_path: Path to the raw data files
            model_path: Path to save/load trained models
        """
        self.data_path = data_path
        self.model_path = model_path
        self.customers_df = None
        self.transactions_df = None
        self.betting_df = None
        self.sessions_df = None
        
        # Trained models storage
        self.models = {
            'churn_prediction': None,
            'ltv_forecasting': None,
            'conversion_prediction': None
        }
        
        # Scalers for feature normalization
        self.scalers = {}
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
    
    def load_data(self):
        """Load all required datasets"""
        try:
            print("Loading data for ML modeling...")
            
            self.customers_df = pd.read_csv(f"{self.data_path}/customers.csv")
            self.customers_df['signup_date'] = pd.to_datetime(self.customers_df['signup_date'])
            
            self.transactions_df = pd.read_csv(f"{self.data_path}/transactions.csv")
            self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
            
            self.betting_df = pd.read_csv(f"{self.data_path}/betting_activity.csv")
            self.betting_df['bet_date'] = pd.to_datetime(self.betting_df['bet_date'])
            
            self.sessions_df = pd.read_csv(f"{self.data_path}/sessions.csv")
            self.sessions_df['start_time'] = pd.to_datetime(self.sessions_df['start_time'])
            
            print(f"  Loaded {len(self.customers_df):,} customers")
            print(f"  Loaded {len(self.betting_df):,} bets")
            print(f"  Loaded {len(self.sessions_df):,} sessions")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run the data generator first: python python/data_generator.py")
    
    def create_feature_matrix(self, prediction_type: str = 'churn') -> pd.DataFrame:
        """
        Create comprehensive feature matrix for ML models
        
        Args:
            prediction_type: 'churn', 'ltv', or 'conversion'
            
        Returns:
            DataFrame with engineered features
        """
        if self.customers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Creating feature matrix for {prediction_type} prediction...")
        
        features_list = []
        
        for _, customer in self.customers_df.iterrows():
            customer_id = customer['customer_id']
            features = self._extract_customer_features(customer_id, prediction_type)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Remove rows with insufficient data
        features_df = features_df.dropna(subset=['customer_id'])
        
        print(f"  Created feature matrix: {features_df.shape}")
        print(f"  Features: {list(features_df.columns)}")
        
        return features_df
    
    def _extract_customer_features(self, customer_id: str, prediction_type: str) -> Dict:
        """
        Extract comprehensive features for a single customer
        
        Args:
            customer_id: Customer identifier
            prediction_type: Type of prediction being made
            
        Returns:
            Dictionary of customer features
        """
        # Get customer basic info
        customer_info = self.customers_df[self.customers_df['customer_id'] == customer_id].iloc[0]
        
        # Current date for calculations (use end of data period)
        current_date = pd.Timestamp('2024-12-31')
        signup_date = customer_info['signup_date']
        
        # Customer demographics and acquisition
        features = {
            'customer_id': customer_id,
            'age': customer_info['age'],
            'acquisition_cost': customer_info['acquisition_cost'],
            'tenure_days': (current_date - signup_date).days,
            'customer_segment_casual': 1 if customer_info['customer_segment'] == 'casual' else 0,
            'customer_segment_regular': 1 if customer_info['customer_segment'] == 'regular' else 0,
            'customer_segment_whale': 1 if customer_info['customer_segment'] == 'whale' else 0,
            'channel_paid_search': 1 if customer_info['acquisition_channel'] == 'paid_search' else 0,
            'channel_social_media': 1 if customer_info['acquisition_channel'] == 'social_media' else 0,
            'channel_affiliate': 1 if customer_info['acquisition_channel'] == 'affiliate' else 0,
            'channel_organic': 1 if customer_info['acquisition_channel'] == 'organic' else 0,
            'email_verified': int(customer_info.get('email_verified', False)),
            'phone_verified': int(customer_info.get('phone_verified', False))
        }
        
        # Transaction features
        customer_transactions = self.transactions_df[
            self.transactions_df['customer_id'] == customer_id
        ]
        
        if len(customer_transactions) > 0:
            deposits = customer_transactions[customer_transactions['transaction_type'] == 'deposit']
            withdrawals = customer_transactions[customer_transactions['transaction_type'] == 'withdrawal']
            
            features.update({
                'total_deposits': deposits['amount'].sum(),
                'total_withdrawals': withdrawals['amount'].sum(),
                'net_deposits': deposits['amount'].sum() - withdrawals['amount'].sum(),
                'num_deposits': len(deposits),
                'num_withdrawals': len(withdrawals),
                'avg_deposit_size': deposits['amount'].mean() if len(deposits) > 0 else 0,
                'first_deposit_amount': deposits['amount'].iloc[0] if len(deposits) > 0 else 0,
                'deposit_frequency': len(deposits) / max(1, features['tenure_days'] / 30),  # per month
                'has_made_deposit': int(len(deposits) > 0)
            })
            
            # Time-based transaction features
            if len(deposits) > 0:
                first_deposit_date = deposits['transaction_date'].min()
                last_transaction_date = customer_transactions['transaction_date'].max()
                
                features.update({
                    'days_to_first_deposit': (first_deposit_date - signup_date).days,
                    'days_since_last_transaction': (current_date - last_transaction_date).days,
                    'transaction_span_days': (last_transaction_date - first_deposit_date).days + 1
                })
            else:
                features.update({
                    'days_to_first_deposit': -1,  # Never deposited
                    'days_since_last_transaction': features['tenure_days'],
                    'transaction_span_days': 0
                })
        else:
            # No transactions
            no_transaction_features = {
                'total_deposits': 0, 'total_withdrawals': 0, 'net_deposits': 0,
                'num_deposits': 0, 'num_withdrawals': 0, 'avg_deposit_size': 0,
                'first_deposit_amount': 0, 'deposit_frequency': 0, 'has_made_deposit': 0,
                'days_to_first_deposit': -1, 'days_since_last_transaction': features['tenure_days'],
                'transaction_span_days': 0
            }
            features.update(no_transaction_features)
        
        # Betting features
        customer_bets = self.betting_df[self.betting_df['customer_id'] == customer_id]
        
        if len(customer_bets) > 0:
            features.update({
                'total_bets': len(customer_bets),
                'total_wagered': customer_bets['bet_amount'].sum(),
                'total_winnings': customer_bets['payout'].sum(),
                'net_betting_result': customer_bets['net_result'].sum(),
                'gross_gaming_revenue': customer_bets['bet_amount'].sum() - customer_bets['payout'].sum(),
                'avg_bet_size': customer_bets['bet_amount'].mean(),
                'bet_size_std': customer_bets['bet_amount'].std(),
                'win_rate': customer_bets['bet_won'].mean(),
                'betting_frequency': len(customer_bets) / max(1, features['tenure_days'] / 7),  # per week
                'sports_diversity': customer_bets['sport'].nunique(),
                'favorite_sport': customer_bets['sport'].mode().iloc[0] if len(customer_bets) > 0 else 'none'
            })
            
            # Time-based betting features
            first_bet_date = customer_bets['bet_date'].min()
            last_bet_date = customer_bets['bet_date'].max()
            
            features.update({
                'days_to_first_bet': (first_bet_date - signup_date).days,
                'days_since_last_bet': (current_date - last_bet_date).days,
                'betting_span_days': (last_bet_date - first_bet_date).days + 1,
                'active_betting_days': customer_bets['bet_date'].dt.date.nunique()
            })
            
            # Recent activity features (last 30 days)
            recent_cutoff = current_date - timedelta(days=30)
            recent_bets = customer_bets[customer_bets['bet_date'] >= recent_cutoff]
            
            features.update({
                'bets_last_30d': len(recent_bets),
                'wagered_last_30d': recent_bets['bet_amount'].sum(),
                'active_days_last_30d': recent_bets['bet_date'].dt.date.nunique()
            })
        else:
            # No betting activity
            no_betting_features = {
                'total_bets': 0, 'total_wagered': 0, 'total_winnings': 0,
                'net_betting_result': 0, 'gross_gaming_revenue': 0,
                'avg_bet_size': 0, 'bet_size_std': 0, 'win_rate': 0,
                'betting_frequency': 0, 'sports_diversity': 0, 'favorite_sport': 'none',
                'days_to_first_bet': -1, 'days_since_last_bet': features['tenure_days'],
                'betting_span_days': 0, 'active_betting_days': 0,
                'bets_last_30d': 0, 'wagered_last_30d': 0, 'active_days_last_30d': 0
            }
            features.update(no_betting_features)
        
        # Session features
        customer_sessions = self.sessions_df[self.sessions_df['customer_id'] == customer_id]
        
        if len(customer_sessions) > 0:
            features.update({
                'total_sessions': len(customer_sessions),
                'avg_session_duration': customer_sessions['duration_minutes'].mean(),
                'total_session_time': customer_sessions['duration_minutes'].sum(),
                'avg_bets_per_session': customer_sessions['num_bets'].mean(),
                'sessions_per_week': len(customer_sessions) / max(1, features['tenure_days'] / 7),
                'mobile_sessions_pct': (customer_sessions['device_type'] == 'mobile').mean(),
                'desktop_sessions_pct': (customer_sessions['device_type'] == 'desktop').mean()
            })
        else:
            # No session data
            no_session_features = {
                'total_sessions': 0, 'avg_session_duration': 0, 'total_session_time': 0,
                'avg_bets_per_session': 0, 'sessions_per_week': 0,
                'mobile_sessions_pct': 0, 'desktop_sessions_pct': 0
            }
            features.update(no_session_features)
        
        # Calculate target variables based on prediction type
        if prediction_type == 'churn':
            # Churn = no activity in last 30 days AND had previous activity
            has_activity = features['total_bets'] > 0 or features['num_deposits'] > 0
            is_churned = (features['days_since_last_bet'] > 30 and 
                         features['days_since_last_transaction'] > 30 and 
                         has_activity)
            features['churn_target'] = int(is_churned)
        
        elif prediction_type == 'ltv':
            # LTV = gross gaming revenue - acquisition cost
            features['ltv_target'] = features['gross_gaming_revenue'] - features['acquisition_cost']
        
        elif prediction_type == 'conversion':
            # Conversion = made first deposit
            features['conversion_target'] = features['has_made_deposit']
        
        return features
    
    def train_churn_model(self, features_df: pd.DataFrame) -> Dict:
        """
        Train churn prediction model
        
        Args:
            features_df: Feature matrix from create_feature_matrix()
            
        Returns:
            Dictionary with model performance metrics
        """
        print("Training churn prediction model...")
        
        # Prepare data
        X = features_df.drop(['customer_id', 'churn_target', 'favorite_sport'], axis=1, errors='ignore')
        y = features_df['churn_target']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['churn'] = scaler
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        best_score = 0
        best_model = None
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            if model_name in ['LogisticRegression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model_name
        
        # Store best model
        self.models['churn_prediction'] = results[best_model]['model']
        
        # Feature importance (for tree-based models)
        if hasattr(self.models['churn_prediction'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['churn_prediction'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance
        
        results['best_model'] = best_model
        results['test_labels'] = y_test
        results['feature_names'] = X.columns.tolist()
        
        print(f"  Best model: {best_model} (AUC: {best_score:.3f})")
        
        return results
    
    def train_ltv_model(self, features_df: pd.DataFrame) -> Dict:
        """
        Train customer lifetime value prediction model
        
        Args:
            features_df: Feature matrix from create_feature_matrix()
            
        Returns:
            Dictionary with model performance metrics
        """
        print("Training LTV prediction model...")
        
        # Only use customers who made deposits (have meaningful LTV)
        ltv_data = features_df[features_df['has_made_deposit'] == 1].copy()
        
        if len(ltv_data) == 0:
            print("  No customers with deposits found!")
            return {}
        
        # Prepare data
        X = ltv_data.drop(['customer_id', 'ltv_target', 'favorite_sport'], axis=1, errors='ignore')
        y = ltv_data['ltv_target']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['ltv'] = scaler
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'Ridge': Ridge(random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        results = {}
        best_score = float('-inf')
        best_model = None
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            if model_name in ['Ridge', 'LinearRegression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'predictions': y_pred
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model_name
        
        # Store best model
        self.models['ltv_forecasting'] = results[best_model]['model']
        
        # Feature importance (for tree-based models)
        if hasattr(self.models['ltv_forecasting'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['ltv_forecasting'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance
        
        results['best_model'] = best_model
        results['test_labels'] = y_test
        results['feature_names'] = X.columns.tolist()
        
        print(f"  Best model: {best_model} (R²: {best_score:.3f})")
        
        return results
    
    def train_conversion_model(self, features_df: pd.DataFrame) -> Dict:
        """
        Train conversion probability prediction model
        
        Args:
            features_df: Feature matrix from create_feature_matrix()
            
        Returns:
            Dictionary with model performance metrics
        """
        print("Training conversion prediction model...")
        
        # Prepare data
        X = features_df.drop(['customer_id', 'conversion_target', 'favorite_sport'], axis=1, errors='ignore')
        y = features_df['conversion_target']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['conversion'] = scaler
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        best_score = 0
        best_model = None
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            if model_name in ['LogisticRegression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model_name
        
        # Store best model
        self.models['conversion_prediction'] = results[best_model]['model']
        
        # Feature importance (for tree-based models)
        if hasattr(self.models['conversion_prediction'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['conversion_prediction'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance
        
        results['best_model'] = best_model
        results['test_labels'] = y_test
        results['feature_names'] = X.columns.tolist()
        
        print(f"  Best model: {best_model} (AUC: {best_score:.3f})")
        
        return results
    
    def save_models(self):
        """Save trained models and scalers"""
        print("Saving models...")
        
        for model_name, model in self.models.items():
            if model is not None:
                joblib.dump(model, f"{self.model_path}/{model_name}.pkl")
                print(f"  Saved {model_name}")
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{self.model_path}/scaler_{scaler_name}.pkl")
            print(f"  Saved scaler_{scaler_name}")
    
    def load_models(self):
        """Load trained models and scalers"""
        print("Loading models...")
        
        for model_name in self.models.keys():
            model_path = f"{self.model_path}/{model_name}.pkl"
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"  Loaded {model_name}")
        
        # Load scalers
        scaler_files = [f for f in os.listdir(self.model_path) if f.startswith('scaler_')]
        for scaler_file in scaler_files:
            scaler_name = scaler_file.replace('scaler_', '').replace('.pkl', '')
            self.scalers[scaler_name] = joblib.load(f"{self.model_path}/{scaler_file}")
            print(f"  Loaded {scaler_name} scaler")
    
    def predict_customer_risk_scores(self, customer_ids: List[str] = None) -> pd.DataFrame:
        """
        Generate risk scores and predictions for customers
        
        Args:
            customer_ids: List of customer IDs to score (None for all)
            
        Returns:
            DataFrame with risk scores and predictions
        """
        if not any(self.models.values()):
            print("No trained models found. Please train models first.")
            return pd.DataFrame()
        
        print("Generating customer risk scores...")
        
        # Create features for scoring
        if customer_ids is None:
            customer_ids = self.customers_df['customer_id'].tolist()
        
        scores = []
        
        for customer_id in customer_ids:
            try:
                # Extract features for this customer
                features = self._extract_customer_features(customer_id, 'churn')  # Use churn features as base
                
                # Convert to DataFrame
                feature_df = pd.DataFrame([features])
                
                # Prepare features for prediction
                X = feature_df.drop(['customer_id', 'churn_target', 'favorite_sport'], axis=1, errors='ignore')
                
                # Handle categorical variables
                categorical_cols = X.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                customer_scores = {'customer_id': customer_id}
                
                # Churn prediction
                if self.models['churn_prediction'] is not None:
                    churn_prob = self.models['churn_prediction'].predict_proba(X)[0][1]
                    customer_scores['churn_probability'] = churn_prob
                    customer_scores['churn_risk'] = 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low'
                
                # LTV prediction (only for customers with deposits)
                if self.models['ltv_forecasting'] is not None and features['has_made_deposit'] == 1:
                    ltv_pred = self.models['ltv_forecasting'].predict(X)[0]
                    customer_scores['predicted_ltv'] = ltv_pred
                    customer_scores['ltv_segment'] = 'High' if ltv_pred > 200 else 'Medium' if ltv_pred > 50 else 'Low'
                else:
                    customer_scores['predicted_ltv'] = 0
                    customer_scores['ltv_segment'] = 'Low'
                
                # Conversion prediction (for non-converted customers)
                if self.models['conversion_prediction'] is not None and features['has_made_deposit'] == 0:
                    conv_prob = self.models['conversion_prediction'].predict_proba(X)[0][1]
                    customer_scores['conversion_probability'] = conv_prob
                    customer_scores['conversion_potential'] = 'High' if conv_prob > 0.7 else 'Medium' if conv_prob > 0.3 else 'Low'
                else:
                    customer_scores['conversion_probability'] = 1.0 if features['has_made_deposit'] else 0.0
                    customer_scores['conversion_potential'] = 'Converted' if features['has_made_deposit'] else 'Low'
                
                # Add some key features for context
                customer_scores.update({
                    'tenure_days': features['tenure_days'],
                    'total_bets': features['total_bets'],
                    'total_wagered': features['total_wagered'],
                    'days_since_last_bet': features['days_since_last_bet'],
                    'customer_segment': self.customers_df[self.customers_df['customer_id'] == customer_id]['customer_segment'].iloc[0]
                })
                
                scores.append(customer_scores)
                
            except Exception as e:
                print(f"Error scoring customer {customer_id}: {e}")
                continue
        
        return pd.DataFrame(scores)
    
    def plot_model_performance(self, model_results: Dict, model_type: str):
        """
        Plot model performance metrics
        
        Args:
            model_results: Results from train_*_model() functions
            model_type: 'churn', 'ltv', or 'conversion'
        """
        if model_type in ['churn', 'conversion']:
            # Classification metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # ROC Curve
            best_model = model_results['best_model']
            y_test = model_results['test_labels']
            y_proba = model_results[best_model]['probabilities']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = model_results[best_model]['auc_score']
            
            axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title(f'{model_type.title()} Model - ROC Curve')
            axes[0, 0].legend()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            axes[0, 1].plot(recall, precision)
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            
            # Feature Importance (if available)
            if 'feature_importance' in model_results:
                feature_imp = model_results['feature_importance'].head(15)
                axes[1, 0].barh(feature_imp['feature'], feature_imp['importance'])
                axes[1, 0].set_title('Top Feature Importances')
                axes[1, 0].set_xlabel('Importance')
            
            # Model Comparison
            model_names = [name for name in model_results.keys() 
                          if isinstance(model_results[name], dict) and 'auc_score' in model_results[name]]
            auc_scores = [model_results[name]['auc_score'] for name in model_names]
            
            axes[1, 1].bar(model_names, auc_scores)
            axes[1, 1].set_title('Model Performance Comparison')
            axes[1, 1].set_ylabel('AUC Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
        else:  # LTV regression
            # Regression metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            best_model = model_results['best_model']
            y_test = model_results['test_labels']
            y_pred = model_results[best_model]['predictions']
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[0, 0].set_xlabel('Actual LTV')
            axes[0, 0].set_ylabel('Predicted LTV')
            axes[0, 0].set_title(f'Actual vs Predicted LTV (R² = {model_results[best_model]["r2_score"]:.3f})')
            
            # Residuals Plot
            residuals = y_test - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted LTV')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Feature Importance (if available)
            if 'feature_importance' in model_results:
                feature_imp = model_results['feature_importance'].head(15)
                axes[1, 0].barh(feature_imp['feature'], feature_imp['importance'])
                axes[1, 0].set_title('Top Feature Importances')
                axes[1, 0].set_xlabel('Importance')
            
            # Model Comparison
            model_names = [name for name in model_results.keys() 
                          if isinstance(model_results[name], dict) and 'r2_score' in model_results[name]]
            r2_scores = [model_results[name]['r2_score'] for name in model_names]
            
            axes[1, 1].bar(model_names, r2_scores)
            axes[1, 1].set_title('Model Performance Comparison')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of ML prediction models"""
    print("🤖 Sportsbook ML Prediction Models")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SportsbookMLPredictor()
    
    # Load data
    predictor.load_data()
    
    if predictor.customers_df is None:
        print("❌ Data not available. Please run data generator first:")
        print("   python python/data_generator.py")
        return
    
    print("\n🔄 Training ML Models...")
    
    # 1. Train churn prediction model
    print("\n1. Churn Prediction Model")
    churn_features = predictor.create_feature_matrix('churn')
    churn_results = predictor.train_churn_model(churn_features)
    
    if churn_results:
        print(f"   Best model: {churn_results['best_model']}")
        print(f"   AUC Score: {churn_results[churn_results['best_model']]['auc_score']:.3f}")
        
        # Show top feature importances
        if 'feature_importance' in churn_results:
            print("\n   Top Churn Prediction Features:")
            print(churn_results['feature_importance'].head(10).to_string(index=False))
    
    # 2. Train LTV prediction model
    print("\n2. Customer LTV Prediction Model")
    ltv_features = predictor.create_feature_matrix('ltv')
    ltv_results = predictor.train_ltv_model(ltv_features)
    
    if ltv_results:
        print(f"   Best model: {ltv_results['best_model']}")
        print(f"   R² Score: {ltv_results[ltv_results['best_model']]['r2_score']:.3f}")
        print(f"   RMSE: ${ltv_results[ltv_results['best_model']]['rmse']:.2f}")
        
        # Show top feature importances
        if 'feature_importance' in ltv_results:
            print("\n   Top LTV Prediction Features:")
            print(ltv_results['feature_importance'].head(10).to_string(index=False))
    
    # 3. Train conversion prediction model
    print("\n3. Conversion Prediction Model")
    conversion_features = predictor.create_feature_matrix('conversion')
    conversion_results = predictor.train_conversion_model(conversion_features)
    
    if conversion_results:
        print(f"   Best model: {conversion_results['best_model']}")
        print(f"   AUC Score: {conversion_results[conversion_results['best_model']]['auc_score']:.3f}")
    
    # 4. Save models
    print("\n💾 Saving Models...")
    predictor.save_models()
    
    # 5. Generate customer risk scores
    print("\n📊 Generating Customer Risk Scores...")
    sample_customers = predictor.customers_df['customer_id'].head(20).tolist()
    risk_scores = predictor.predict_customer_risk_scores(sample_customers)
    
    if len(risk_scores) > 0:
        print("\n   Sample Risk Scores:")
        display_cols = ['customer_id', 'churn_probability', 'churn_risk', 
                       'predicted_ltv', 'conversion_probability', 'customer_segment']
        available_cols = [col for col in display_cols if col in risk_scores.columns]
        print(risk_scores[available_cols].head(10).to_string(index=False))
    
    # 6. Generate visualizations
    print("\n📈 Generating Model Performance Plots...")
    
    if churn_results:
        predictor.plot_model_performance(churn_results, 'churn')
    
    if ltv_results:
        predictor.plot_model_performance(ltv_results, 'ltv')
    
    # 7. Business insights
    print("\n💡 Business Insights:")
    
    if len(risk_scores) > 0:
        high_churn_risk = len(risk_scores[risk_scores['churn_risk'] == 'High'])
        high_ltv_customers = len(risk_scores[risk_scores['ltv_segment'] == 'High'])
        high_conversion_potential = len(risk_scores[risk_scores['conversion_potential'] == 'High'])
        
        print(f"   High churn risk customers: {high_churn_risk}")
        print(f"   High LTV customers: {high_ltv_customers}")
        print(f"   High conversion potential: {high_conversion_potential}")
        
        avg_predicted_ltv = risk_scores['predicted_ltv'].mean()
        print(f"   Average predicted LTV: ${avg_predicted_ltv:.2f}")
    
    print("\n✅ ML modeling complete!")
    print("\nNext steps:")
    print("  1. Use churn predictions for retention campaigns")
    print("  2. Target high-LTV customers with premium offers")
    print("  3. Focus conversion efforts on high-probability prospects")
    print("  4. Monitor model performance and retrain regularly")

if __name__ == "__main__":
    main()