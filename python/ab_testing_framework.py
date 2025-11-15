"""
A/B Testing Framework for Sportsbook Marketing Campaigns

This module provides comprehensive tools for designing, implementing,
and analyzing A/B tests for marketing campaigns, including statistical
significance testing, confidence intervals, and power analysis.

Author: Sportsbook Marketing Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class ABTestResult:
    """Container for A/B test results"""
    metric_name: str
    control_value: float
    treatment_value: float
    lift: float
    lift_pct: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    statistical_power: float
    effect_size: float
    sample_size_control: int
    sample_size_treatment: int

class ABTestAnalyzer:
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize A/B Test Analyzer
        
        Args:
            alpha: Significance level (Type I error rate)
            power: Desired statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.power = power
        self.confidence_level = 1 - alpha
    
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        expected_lift: float, 
        test_type: str = 'proportion'
    ) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Current conversion rate or baseline metric
            expected_lift: Expected improvement (as decimal, e.g., 0.1 for 10% lift)
            test_type: 'proportion' or 'continuous'
        
        Returns:
            Required sample size per group
        """
        if test_type == 'proportion':
            # For proportions (conversion rates)
            p1 = baseline_rate
            p2 = baseline_rate * (1 + expected_lift)
            
            # Pooled proportion
            p_pooled = (p1 + p2) / 2
            
            # Effect size (Cohen's h for proportions)
            effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
            
            # Z-scores for alpha and power
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(self.power)
            
            # Sample size calculation
            n = ((z_alpha + z_beta) ** 2 * p_pooled * (1 - p_pooled)) / ((p2 - p1) ** 2)
            
        else:  # continuous metrics
            # For continuous metrics, assume we need effect size
            effect_size = expected_lift  # Assume this is Cohen's d
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(self.power)
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def analyze_conversion_rate(
        self, 
        control_conversions: int, 
        control_total: int,
        treatment_conversions: int, 
        treatment_total: int
    ) -> ABTestResult:
        """
        Analyze A/B test for conversion rate metrics
        
        Args:
            control_conversions: Number of conversions in control group
            control_total: Total number in control group
            treatment_conversions: Number of conversions in treatment group  
            treatment_total: Total number in treatment group
            
        Returns:
            ABTestResult object with test results
        """
        # Calculate conversion rates
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        
        # Calculate lift
        lift = treatment_rate - control_rate
        lift_pct = (lift / control_rate) * 100 if control_rate > 0 else 0
        
        # Chi-square test for independence
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions]
        ])
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Confidence interval for difference in proportions
        se_diff = np.sqrt(
            (control_rate * (1 - control_rate) / control_total) + 
            (treatment_rate * (1 - treatment_rate) / treatment_total)
        )
        
        z_score = stats.norm.ppf(1 - self.alpha/2)
        margin_error = z_score * se_diff
        ci_lower = lift - margin_error
        ci_upper = lift + margin_error
        
        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
        
        # Statistical power (post-hoc)
        pooled_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se_pooled = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
        z_observed = abs(lift) / se_pooled if se_pooled > 0 else 0
        power_observed = stats.norm.cdf(z_observed - stats.norm.ppf(1 - self.alpha/2)) + \
                        stats.norm.cdf(-z_observed - stats.norm.ppf(1 - self.alpha/2))
        
        return ABTestResult(
            metric_name="Conversion Rate",
            control_value=control_rate,
            treatment_value=treatment_rate,
            lift=lift,
            lift_pct=lift_pct,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            statistical_power=power_observed,
            effect_size=effect_size,
            sample_size_control=control_total,
            sample_size_treatment=treatment_total
        )
    
    def analyze_continuous_metric(
        self,
        control_values: List[float],
        treatment_values: List[float],
        metric_name: str = "Continuous Metric"
    ) -> ABTestResult:
        """
        Analyze A/B test for continuous metrics (e.g., average bet size, LTV)
        
        Args:
            control_values: List of metric values for control group
            treatment_values: List of metric values for treatment group
            metric_name: Name of the metric being tested
            
        Returns:
            ABTestResult object with test results
        """
        control_array = np.array(control_values)
        treatment_array = np.array(treatment_values)
        
        # Calculate means
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        
        # Calculate lift
        lift = treatment_mean - control_mean
        lift_pct = (lift / control_mean) * 100 if control_mean > 0 else 0
        
        # Check for normality (Shapiro-Wilk test on small samples)
        if len(control_values) < 5000 and len(treatment_values) < 5000:
            _, p_control_norm = stats.shapiro(control_array)
            _, p_treatment_norm = stats.shapiro(treatment_array)
            is_normal = (p_control_norm > 0.05) and (p_treatment_norm > 0.05)
        else:
            # Assume normal for large samples due to CLT
            is_normal = True
        
        if is_normal:
            # Use t-test for normal distributions
            t_stat, p_value = ttest_ind(treatment_array, control_array, equal_var=False)
            
            # Confidence interval for difference in means
            se_diff = np.sqrt(
                np.var(control_array, ddof=1) / len(control_array) +
                np.var(treatment_array, ddof=1) / len(treatment_array)
            )
            
            df = len(control_array) + len(treatment_array) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_error = t_critical * se_diff
            
        else:
            # Use Mann-Whitney U test for non-normal distributions
            u_stat, p_value = mannwhitneyu(
                treatment_array, control_array, alternative='two-sided'
            )
            
            # Confidence interval using bootstrap
            margin_error = self._bootstrap_ci_diff(control_array, treatment_array)
        
        ci_lower = lift - margin_error
        ci_upper = lift + margin_error
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_array) - 1) * np.var(control_array, ddof=1) +
             (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)) /
            (len(control_array) + len(treatment_array) - 2)
        )
        effect_size = lift / pooled_std if pooled_std > 0 else 0
        
        # Statistical power (approximate)
        power_observed = self._calculate_power_continuous(
            control_array, treatment_array, effect_size
        )
        
        return ABTestResult(
            metric_name=metric_name,
            control_value=control_mean,
            treatment_value=treatment_mean,
            lift=lift,
            lift_pct=lift_pct,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            statistical_power=power_observed,
            effect_size=effect_size,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values)
        )
    
    def _bootstrap_ci_diff(
        self, 
        control: np.ndarray, 
        treatment: np.ndarray, 
        n_bootstrap: int = 1000
    ) -> float:
        """Calculate confidence interval using bootstrap method"""
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            diff = np.mean(treatment_sample) - np.mean(control_sample)
            bootstrap_diffs.append(diff)
        
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha/2) * 100)
        
        return (ci_upper - ci_lower) / 2  # Return margin of error
    
    def _calculate_power_continuous(
        self, 
        control: np.ndarray, 
        treatment: np.ndarray, 
        effect_size: float
    ) -> float:
        """Calculate statistical power for continuous metrics"""
        n_control = len(control)
        n_treatment = len(treatment)
        
        # Use Cohen's formula for power calculation
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(self.power)
        
        # Effective sample size
        n_eff = (n_control * n_treatment) / (n_control + n_treatment)
        
        # Critical effect size for current sample
        critical_effect = z_alpha / np.sqrt(n_eff / 2)
        
        # Actual power
        if abs(effect_size) > critical_effect:
            power = stats.norm.cdf(np.sqrt(n_eff / 2) * abs(effect_size) - z_alpha)
        else:
            power = self.alpha  # No power if effect is too small
            
        return min(1.0, max(0.0, power))

    def run_campaign_analysis(
        self, 
        campaign_data: pd.DataFrame,
        control_group: str = 'control',
        treatment_group: str = 'treatment'
    ) -> Dict[str, ABTestResult]:
        """
        Run comprehensive A/B test analysis on campaign data
        
        Args:
            campaign_data: DataFrame with campaign results
            control_group: Name of control group in 'group' column
            treatment_group: Name of treatment group in 'group' column
            
        Returns:
            Dictionary of test results for different metrics
        """
        results = {}
        
        control_df = campaign_data[campaign_data['group'] == control_group]
        treatment_df = campaign_data[campaign_data['group'] == treatment_group]
        
        # 1. Response Rate Analysis
        if 'responded' in campaign_data.columns:
            control_responses = control_df['responded'].sum()
            control_total = len(control_df)
            treatment_responses = treatment_df['responded'].sum()
            treatment_total = len(treatment_df)
            
            results['response_rate'] = self.analyze_conversion_rate(
                control_responses, control_total,
                treatment_responses, treatment_total
            )
        
        # 2. Conversion Rate Analysis (first deposit)
        if 'converted' in campaign_data.columns:
            control_conversions = control_df['converted'].sum()
            control_total = len(control_df)
            treatment_conversions = treatment_df['converted'].sum()
            treatment_total = len(treatment_df)
            
            results['conversion_rate'] = self.analyze_conversion_rate(
                control_conversions, control_total,
                treatment_conversions, treatment_total
            )
        
        # 3. Average Bet Size Analysis
        if 'avg_bet_size' in campaign_data.columns:
            control_bets = control_df['avg_bet_size'].dropna().tolist()
            treatment_bets = treatment_df['avg_bet_size'].dropna().tolist()
            
            if len(control_bets) > 0 and len(treatment_bets) > 0:
                results['avg_bet_size'] = self.analyze_continuous_metric(
                    control_bets, treatment_bets, "Average Bet Size"
                )
        
        # 4. Customer Lifetime Value Analysis
        if 'customer_ltv' in campaign_data.columns:
            control_ltv = control_df['customer_ltv'].dropna().tolist()
            treatment_ltv = treatment_df['customer_ltv'].dropna().tolist()
            
            if len(control_ltv) > 0 and len(treatment_ltv) > 0:
                results['customer_ltv'] = self.analyze_continuous_metric(
                    control_ltv, treatment_ltv, "Customer LTV"
                )
        
        # 5. Total Deposits Analysis
        if 'total_deposits' in campaign_data.columns:
            control_deposits = control_df['total_deposits'].dropna().tolist()
            treatment_deposits = treatment_df['total_deposits'].dropna().tolist()
            
            if len(control_deposits) > 0 and len(treatment_deposits) > 0:
                results['total_deposits'] = self.analyze_continuous_metric(
                    control_deposits, treatment_deposits, "Total Deposits"
                )
        
        return results
    
    def generate_test_report(self, results: Dict[str, ABTestResult]) -> pd.DataFrame:
        """
        Generate a summary report of all A/B test results
        
        Args:
            results: Dictionary of ABTestResult objects
            
        Returns:
            DataFrame with test summary
        """
        report_data = []
        
        for metric_name, result in results.items():
            report_data.append({
                'Metric': result.metric_name,
                'Control': f"{result.control_value:.4f}",
                'Treatment': f"{result.treatment_value:.4f}",
                'Lift': f"{result.lift:.4f}",
                'Lift %': f"{result.lift_pct:.2f}%",
                'P-Value': f"{result.p_value:.4f}",
                'Significant': "✓" if result.is_significant else "✗",
                'CI Lower': f"{result.confidence_interval[0]:.4f}",
                'CI Upper': f"{result.confidence_interval[1]:.4f}",
                'Effect Size': f"{result.effect_size:.4f}",
                'Statistical Power': f"{result.statistical_power:.2f}",
                'Sample Size (C)': result.sample_size_control,
                'Sample Size (T)': result.sample_size_treatment
            })
        
        return pd.DataFrame(report_data)

class CampaignSimulator:
    """Simulate A/B test campaigns for different scenarios"""
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
    
    def simulate_welcome_bonus_test(
        self,
        n_control: int = 1000,
        n_treatment: int = 1000,
        control_conversion_rate: float = 0.08,
        treatment_lift: float = 0.15,  # 15% lift
        control_avg_deposit: float = 75.0,
        treatment_deposit_lift: float = 0.20  # 20% higher deposits
    ) -> pd.DataFrame:
        """
        Simulate a welcome bonus A/B test
        
        Args:
            n_control: Number of users in control group
            n_treatment: Number of users in treatment group
            control_conversion_rate: Baseline conversion rate
            treatment_lift: Expected lift in conversion rate
            control_avg_deposit: Average first deposit in control
            treatment_deposit_lift: Lift in deposit amounts
            
        Returns:
            DataFrame with simulated campaign data
        """
        data = []
        
        # Control group
        for i in range(n_control):
            # Response to campaign (email open/click)
            responded = np.random.random() < 0.25  # 25% response rate
            
            # Conversion (first deposit)
            converted = np.random.random() < control_conversion_rate
            
            # First deposit amount (if converted)
            if converted:
                first_deposit = max(0, np.random.normal(control_avg_deposit, 25))
                total_deposits = first_deposit + np.random.exponential(50)  # Additional deposits
            else:
                first_deposit = 0
                total_deposits = 0
            
            # Betting behavior
            if converted:
                num_bets = np.random.poisson(8)  # Average 8 bets
                avg_bet_size = max(5, np.random.normal(25, 8))
                total_wagered = num_bets * avg_bet_size
                
                # Customer LTV (simplified)
                house_edge = 0.05  # 5% house edge
                customer_ltv = total_wagered * house_edge
            else:
                num_bets = 0
                avg_bet_size = 0
                total_wagered = 0
                customer_ltv = 0
            
            data.append({
                'customer_id': f"CTRL_{i:06d}",
                'group': 'control',
                'responded': responded,
                'converted': converted,
                'first_deposit': first_deposit,
                'total_deposits': total_deposits,
                'num_bets': num_bets,
                'avg_bet_size': avg_bet_size,
                'total_wagered': total_wagered,
                'customer_ltv': customer_ltv
            })
        
        # Treatment group (with bonus)
        treatment_conversion_rate = control_conversion_rate * (1 + treatment_lift)
        treatment_avg_deposit = control_avg_deposit * (1 + treatment_deposit_lift)
        
        for i in range(n_treatment):
            # Higher response rate due to attractive bonus
            responded = np.random.random() < 0.35  # 35% response rate
            
            # Higher conversion rate
            converted = np.random.random() < treatment_conversion_rate
            
            # First deposit amount (if converted)
            if converted:
                first_deposit = max(0, np.random.normal(treatment_avg_deposit, 25))
                # Bonus might encourage more deposits
                total_deposits = first_deposit + np.random.exponential(65)
            else:
                first_deposit = 0
                total_deposits = 0
            
            # Betting behavior (bonus users might bet more)
            if converted:
                num_bets = np.random.poisson(12)  # More bets due to bonus
                avg_bet_size = max(5, np.random.normal(28, 10))  # Slightly higher bets
                total_wagered = num_bets * avg_bet_size
                
                # Customer LTV
                house_edge = 0.05
                customer_ltv = total_wagered * house_edge
            else:
                num_bets = 0
                avg_bet_size = 0
                total_wagered = 0
                customer_ltv = 0
            
            data.append({
                'customer_id': f"TRTM_{i:06d}",
                'group': 'treatment',
                'responded': responded,
                'converted': converted,
                'first_deposit': first_deposit,
                'total_deposits': total_deposits,
                'num_bets': num_bets,
                'avg_bet_size': avg_bet_size,
                'total_wagered': total_wagered,
                'customer_ltv': customer_ltv
            })
        
        return pd.DataFrame(data)
    
    def simulate_retention_campaign(
        self,
        n_control: int = 500,
        n_treatment: int = 500,
        baseline_reactivation_rate: float = 0.12,
        treatment_lift: float = 0.25
    ) -> pd.DataFrame:
        """Simulate a customer retention/reactivation campaign"""
        
        data = []
        treatment_reactivation_rate = baseline_reactivation_rate * (1 + treatment_lift)
        
        # Control group (no special offer)
        for i in range(n_control):
            responded = np.random.random() < 0.15  # Low response to generic message
            reactivated = np.random.random() < baseline_reactivation_rate
            
            if reactivated:
                bets_after_campaign = np.random.poisson(3)
                avg_bet_size = max(5, np.random.normal(20, 5))
                total_wagered = bets_after_campaign * avg_bet_size
            else:
                bets_after_campaign = 0
                avg_bet_size = 0
                total_wagered = 0
            
            data.append({
                'customer_id': f"RET_CTRL_{i:04d}",
                'group': 'control',
                'responded': responded,
                'reactivated': reactivated,
                'bets_after_campaign': bets_after_campaign,
                'avg_bet_size': avg_bet_size,
                'total_wagered_post_campaign': total_wagered
            })
        
        # Treatment group (special reactivation offer)
        for i in range(n_treatment):
            responded = np.random.random() < 0.28  # Higher response to special offer
            reactivated = np.random.random() < treatment_reactivation_rate
            
            if reactivated:
                bets_after_campaign = np.random.poisson(5)  # More activity due to offer
                avg_bet_size = max(5, np.random.normal(22, 6))
                total_wagered = bets_after_campaign * avg_bet_size
            else:
                bets_after_campaign = 0
                avg_bet_size = 0
                total_wagered = 0
            
            data.append({
                'customer_id': f"RET_TRTM_{i:04d}",
                'group': 'treatment',
                'responded': responded,
                'reactivated': reactivated,
                'bets_after_campaign': bets_after_campaign,
                'avg_bet_size': avg_bet_size,
                'total_wagered_post_campaign': total_wagered
            })
        
        return pd.DataFrame(data)

class ABTestVisualizer:
    """Create visualizations for A/B test results"""
    
    @staticmethod
    def plot_test_results(results: Dict[str, ABTestResult], figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive visualization of A/B test results
        
        Args:
            results: Dictionary of ABTestResult objects
            figsize: Figure size for matplotlib plots
        """
        n_metrics = len(results)
        if n_metrics == 0:
            print("No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Lift comparison
        metrics = list(results.keys())
        lifts = [results[m].lift_pct for m in metrics]
        p_values = [results[m].p_value for m in metrics]
        
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        
        axes[0].barh(metrics, lifts, color=colors, alpha=0.7)
        axes[0].set_xlabel('Lift (%)')
        axes[0].set_title('Treatment Lift by Metric')
        axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add significance markers
        for i, (metric, lift) in enumerate(zip(metrics, lifts)):
            is_sig = p_values[i] < 0.05
            marker = '***' if is_sig else 'n.s.'
            axes[0].text(lift + (max(lifts) * 0.02), i, marker, 
                        verticalalignment='center', fontweight='bold')
        
        # 2. P-values with significance threshold
        axes[1].barh(metrics, p_values, alpha=0.7)
        axes[1].set_xlabel('P-Value')
        axes[1].set_title('Statistical Significance')
        axes[1].axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        axes[1].legend()
        
        # 3. Confidence intervals
        for i, metric in enumerate(metrics):
            result = results[metric]
            ci_lower, ci_upper = result.confidence_interval
            lift_pct = result.lift_pct
            
            # Plot point estimate and CI
            axes[2].scatter(lift_pct, i, s=100, alpha=0.8, 
                           color='green' if result.is_significant else 'red')
            axes[2].plot([ci_lower*100, ci_upper*100], [i, i], 'k-', alpha=0.6)
            axes[2].plot([ci_lower*100, ci_lower*100], [i-0.1, i+0.1], 'k-', alpha=0.6)
            axes[2].plot([ci_upper*100, ci_upper*100], [i-0.1, i+0.1], 'k-', alpha=0.6)
        
        axes[2].set_yticks(range(len(metrics)))
        axes[2].set_yticklabels(metrics)
        axes[2].set_xlabel('Lift % (with 95% CI)')
        axes[2].set_title('Lift with Confidence Intervals')
        axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Effect sizes
        effect_sizes = [results[m].effect_size for m in metrics]
        axes[3].barh(metrics, effect_sizes, alpha=0.7)
        axes[3].set_xlabel('Effect Size (Cohen\'s d/h)')
        axes[3].set_title('Effect Sizes')
        
        # Add effect size interpretation
        axes[3].axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small')
        axes[3].axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Medium')
        axes[3].axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
        axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_interactive_dashboard(results: Dict[str, ABTestResult]) -> go.Figure:
        """
        Create interactive Plotly dashboard for A/B test results
        
        Args:
            results: Dictionary of ABTestResult objects
            
        Returns:
            Plotly figure object
        """
        if not results:
            return go.Figure()
        
        metrics = list(results.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Treatment Lift (%)', 'Statistical Significance', 
                          'Confidence Intervals', 'Sample Sizes'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Lift comparison
        lifts = [results[m].lift_pct for m in metrics]
        p_values = [results[m].p_value for m in metrics]
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        
        fig.add_trace(
            go.Bar(
                y=metrics,
                x=lifts,
                orientation='h',
                marker=dict(color=colors),
                name='Lift %',
                text=[f"{l:.1f}%" for l in lifts],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. P-values
        fig.add_trace(
            go.Bar(
                y=metrics,
                x=p_values,
                orientation='h',
                marker=dict(color=['lightcoral' if p > 0.05 else 'lightgreen' for p in p_values]),
                name='P-Value',
                text=[f"{p:.4f}" for p in p_values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Add significance line
        fig.add_shape(
            type="line",
            x0=0.05, x1=0.05,
            y0=-0.5, y1=len(metrics)-0.5,
            line=dict(color="red", dash="dash"),
            row=1, col=2
        )
        
        # 3. Confidence intervals
        for i, metric in enumerate(metrics):
            result = results[metric]
            ci_lower, ci_upper = result.confidence_interval
            lift_pct = result.lift_pct
            
            # Point estimate
            fig.add_trace(
                go.Scatter(
                    x=[lift_pct],
                    y=[i],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='green' if result.is_significant else 'red'
                    ),
                    name=f'{metric} Lift',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=[ci_lower*100, ci_upper*100],
                    y=[i, i],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name=f'{metric} CI',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Sample sizes
        control_sizes = [results[m].sample_size_control for m in metrics]
        treatment_sizes = [results[m].sample_size_treatment for m in metrics]
        
        fig.add_trace(
            go.Bar(
                y=metrics,
                x=control_sizes,
                orientation='h',
                name='Control',
                marker=dict(color='lightblue')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                y=metrics,
                x=treatment_sizes,
                orientation='h',
                name='Treatment',
                marker=dict(color='orange')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="A/B Test Results Dashboard",
            showlegend=True
        )
        
        return fig

def main():
    """Example usage of A/B testing framework"""
    print("🧪 Sportsbook A/B Testing Framework")
    print("=" * 50)
    
    # Initialize analyzer
    ab_test = ABTestAnalyzer(alpha=0.05, power=0.8)
    simulator = CampaignSimulator()
    
    # 1. Sample size calculation example
    print("\n📊 Sample Size Calculation")
    baseline_conversion = 0.08  # 8% baseline conversion
    expected_lift = 0.15  # Expect 15% improvement
    
    required_n = ab_test.calculate_sample_size(
        baseline_conversion, expected_lift, 'proportion'
    )
    print(f"Required sample size per group: {required_n:,}")
    
    # 2. Simulate welcome bonus campaign
    print(f"\n🎁 Simulating Welcome Bonus Campaign (n={required_n:,} per group)")
    campaign_data = simulator.simulate_welcome_bonus_test(
        n_control=required_n,
        n_treatment=required_n,
        control_conversion_rate=baseline_conversion,
        treatment_lift=expected_lift
    )
    
    # 3. Analyze results
    print("\n📈 Analyzing Campaign Results")
    results = ab_test.run_campaign_analysis(campaign_data)
    
    # 4. Generate report
    report_df = ab_test.generate_test_report(results)
    print("\n📋 A/B Test Summary Report:")
    print(report_df.to_string(index=False))
    
    # 5. Create visualizations
    print("\n📊 Generating Visualizations...")
    ABTestVisualizer.plot_test_results(results)
    
    # 6. Calculate ROI
    if 'customer_ltv' in results:
        ltv_result = results['customer_ltv']
        ltv_lift = ltv_result.lift
        campaign_cost = 15  # $15 per customer campaign cost
        
        roi = (ltv_lift - campaign_cost) / campaign_cost * 100
        print(f"\n💰 Campaign ROI Analysis:")
        print(f"  LTV Lift: ${ltv_lift:.2f}")
        print(f"  Campaign Cost: ${campaign_cost:.2f}")
        print(f"  ROI: {roi:.1f}%")
    
    print("\n✅ A/B testing framework demonstration complete!")

if __name__ == "__main__":
    main()