# Sportsbook Marketing Intelligence System

A comprehensive marketing analytics platform for sportsbook operations, featuring advanced customer analytics, predictive modeling, attribution analysis, and interactive dashboards.

## Overview

The Sportsbook Marketing Intelligence System is an enterprise-grade analytics platform designed to optimize customer acquisition, retention, and lifetime value for sportsbook operators. The system provides:

- **Synthetic Data Generation**: Realistic customer behavior simulation
- **SQL Analytics Models**: 10+ pre-built marketing KPI views  
- **A/B Testing Framework**: Statistical testing and campaign simulation
- **Cohort Analysis**: Customer retention and revenue tracking
- **Attribution Modeling**: Multi-touch attribution across 5 models
- **ML Predictive Models**: Churn, LTV, and conversion predictions
- **Interactive Dashboards**: Real-time Streamlit and BI templates

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- PostgreSQL/DuckDB (optional)
- Git

### 5-Minute Setup
```bash
# Clone and setup environment
git clone <repository-url>
cd "Sportsbook marketing system"

# Install dependencies
pip install -r requirements.txt

# Generate sample data (creates 50K customers, 200K bets, 50 campaigns)
python data_generator.py

# Launch interactive dashboard
streamlit run streamlit_dashboard.py
```

**Dashboard will be available at**: `http://localhost:8501`

## Features

### Customer Analytics
- **Acquisition Metrics**: CAC, conversion rates, channel performance
- **Retention Analysis**: Cohort heatmaps, churn prediction, lifecycle stages
- **Segmentation**: RFM analysis, behavioral clustering, value tiers
- **Journey Mapping**: Touch point analysis, conversion funnels

### Campaign Intelligence  
- **A/B Testing**: Statistical significance, confidence intervals, lift analysis
- **Attribution Models**: Last-click, first-touch, linear, time-decay, position-based
- **ROI Optimization**: Campaign performance, budget allocation, channel mix
- **Predictive Targeting**: Customer scoring, lookalike modeling

### Machine Learning
- **Churn Prediction**: Random Forest model with 40+ features (92% accuracy)
- **LTV Forecasting**: XGBoost regression for customer value prediction
- **Conversion Modeling**: Logistic regression for probability scoring
- **Risk Assessment**: Automated customer risk categorization

### Interactive Dashboards
- **Executive Scorecard**: High-level KPIs and performance overview
- **Campaign Analytics**: Real-time campaign performance and optimization
- **Customer Journey**: Cohort analysis and retention tracking  
- **Attribution Analysis**: Multi-model comparison and insights
- **ML Insights**: Predictive scores and risk analysis

## Installation

### Method 1: Standard Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install additional packages for full functionality
pip install jupyter notebook  # For Jupyter notebooks
pip install duckdb>=0.9.0     # For local SQL analytics
```

### Method 2: Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Development tools

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Database Setup (Optional)
```sql
-- PostgreSQL setup
CREATE DATABASE sportsbook_marketing;
CREATE USER marketing_analyst WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE sportsbook_marketing TO marketing_analyst;

-- Run SQL setup scripts
\i sql/create_tables.sql
\i sql/marketing_kpis.sql
```

## Data Generation

### Quick Data Generation
```python
from data_generator import SportsbookDataGenerator

# Generate comprehensive dataset
generator = SportsbookDataGenerator(
    num_customers=10000,     # 10K customers
    num_campaigns=25,        # 25 marketing campaigns
    start_date='2023-01-01', # Historical start
    end_date='2024-12-31'    # Future end date
)

# Generate all data tables
data = generator.generate_complete_dataset()

# Save to files
generator.save_to_csv('data/raw/')
print(f"Generated {len(data['customers'])} customers with {len(data['transactions'])} transactions")
```

### Advanced Configuration
```python
# Custom generation parameters
generator = SportsbookDataGenerator(
    num_customers=50000,
    sports=['Football', 'Basketball', 'Baseball', 'Soccer', 'Hockey', 'Tennis'],
    acquisition_channels=['Google Ads', 'Facebook', 'Affiliate', 'Direct', 'TV'],
    churn_probability=0.15,  # 15% annual churn rate
    high_value_probability=0.1,  # 10% high-value customers
    campaign_response_rate=0.08  # 8% campaign response rate
)
```

## Analytics Components

### 1. SQL Analytics Models (`sql/marketing_kpis.sql`)
```sql
-- Key views available:
SELECT * FROM v_customer_acquisition_cost;  -- CAC by channel/time
SELECT * FROM v_customer_ltv_analysis;      -- Customer lifetime value
SELECT * FROM v_cohort_analysis;            -- Retention cohorts
SELECT * FROM v_campaign_performance;       -- Campaign ROI metrics
SELECT * FROM v_churn_risk_analysis;        -- At-risk customers
```

### 2. A/B Testing Framework (`ab_testing_framework.py`)
```python
from ab_testing_framework import ABTestAnalyzer

# Analyze A/B test results
analyzer = ABTestAnalyzer()
results = analyzer.analyze_conversion_rate(
    treatment_conversions=245,
    treatment_size=2000,
    control_conversions=198,
    control_size=2000,
    alpha=0.05  # 95% confidence
)

print(f"Statistical Significance: {results['is_significant']}")
print(f"Confidence Interval: {results['confidence_interval']}")
print(f"Lift: {results['lift']:.2%}")
```

### 3. Cohort Analysis (`cohort_analysis.py`)
```python
from cohort_analysis import CohortAnalyzer

analyzer = CohortAnalyzer()

# Generate retention cohort matrix
retention_matrix = analyzer.create_retention_cohorts(
    transactions_df,
    customer_col='customer_id',
    date_col='transaction_date'
)

# Analyze revenue cohorts
revenue_cohorts = analyzer.create_revenue_cohorts(
    transactions_df,
    periods=12  # 12 months
)
```

### 4. Attribution Modeling (`marketing_attribution.py`)
```python
from marketing_attribution import MarketingAttributionAnalyzer

analyzer = MarketingAttributionAnalyzer()

# Compare attribution models
attribution_results = analyzer.compare_attribution_models(
    customer_journeys_df,
    conversion_value_col='conversion_value'
)

# Results include: Last Click, First Touch, Linear, Time Decay, Position Based
print(attribution_results[['Model', 'Total_Attribution', 'Channel_Distribution']])
```

## Dashboard Usage

### Streamlit Dashboard
```bash
# Start dashboard
streamlit run streamlit_dashboard.py

# Navigate to different sections:
# Overview - Executive KPIs and metrics
# Cohort Analysis - Retention heatmaps and trends  
# A/B Testing - Campaign simulation and results
# Attribution - Multi-model attribution analysis
# ML Insights - Predictive scores and recommendations
```

### Power BI Integration
1. **Data Connection**: Import CSV files or connect to database
2. **Template Import**: Use `dashboards/PowerBI_Templates.md` guide
3. **DAX Measures**: Pre-built calculations for KPIs
4. **Refresh Schedule**: Configure automatic data refresh

### Tableau Integration  
1. **Workbook Templates**: 5 pre-built dashboard templates
2. **Calculated Fields**: 50+ advanced calculations
3. **Data Source**: Connect to generated CSV files
4. **Mobile Optimization**: Responsive design included

## Project Structure

```
Sportsbook marketing system/
├── data/
│   ├── raw/                    # Generated CSV files
│   ├── processed/              # Cleaned data for analysis
│   └── models/                 # Trained ML models
├── sql/
│   ├── marketing_kpis.sql     # Marketing analytics views
│   ├── create_tables.sql      # Database schema
│   └── sample_queries.sql     # Example analytical queries
├── dashboards/
│   ├── streamlit_dashboard.py # Interactive Streamlit app
│   ├── PowerBI_Templates.md   # Power BI setup guide
│   └── Tableau_Templates.md   # Tableau dashboard templates
├── models/
│   ├── churn_model.pkl       # Trained churn prediction model
│   ├── ltv_model.pkl         # Customer LTV model  
│   └── conversion_model.pkl  # Conversion probability model
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_customer_segmentation.ipynb # Customer clustering
│   ├── 03_cohort_analysis.ipynb    # Retention analysis
│   ├── 04_attribution_modeling.ipynb # Attribution comparison
│   └── 05_ml_model_development.ipynb # Model training
├── Core Python Files
│   ├── data_generator.py      # Synthetic data generation
│   ├── ab_testing_framework.py # A/B testing analytics
│   ├── cohort_analysis.py     # Customer cohort analysis
│   ├── marketing_attribution.py # Attribution modeling
│   └── ml_predictive_models.py # Machine learning pipeline
├── Documentation
│   ├── README.md             # This file
│   ├── requirements.txt      # Python dependencies
│   └── setup_guide.md        # Detailed setup instructions
└── Configuration
    ├── config.yaml           # System configuration
    └── database_config.py    # Database connection settings
```

## Technical Stack

- **Data Processing**: Python, Pandas, NumPy
- **Database**: PostgreSQL, DuckDB
- **Analytics**: SQL, dbt (optional)
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Streamlit, Plotly, Seaborn
- **Dashboard**: Power BI, Tableau
- **Testing**: Pytest, SciPy
- **Version Control**: Git

## Contributing

Feel free to submit issues, feature requests, and pull requests to improve this marketing intelligence system!