# 🏆 Sportsbook Marketing Intelligence System

A comprehensive marketing analytics platform for sportsbook operations, featuring advanced customer analytics, predictive modeling, attribution analysis, and interactive dashboards.

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [📊 Features](#-features)
- [🛠 Installation](#-installation)
- [💾 Data Generation](#-data-generation)
- [🔍 Analytics Components](#-analytics-components)
- [📈 Dashboard Usage](#-dashboard-usage)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [📞 Support & Contributing](#-support--contributing)

## 🎯 Overview

The Sportsbook Marketing Intelligence System is an enterprise-grade analytics platform designed to optimize customer acquisition, retention, and lifetime value for sportsbook operators. The system provides:

- **Synthetic Data Generation**: Realistic customer behavior simulation
- **SQL Analytics Models**: 10+ pre-built marketing KPI views  
- **A/B Testing Framework**: Statistical testing and campaign simulation
- **Cohort Analysis**: Customer retention and revenue tracking
- **Attribution Modeling**: Multi-touch attribution across 5 models
- **ML Predictive Models**: Churn, LTV, and conversion predictions
- **Interactive Dashboards**: Real-time Streamlit and BI templates

## 🚀 Quick Start

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

## 🏗️ Architecture

```
📁 Sportsbook Marketing System/
├── 📁 data/                    # Data storage
│   ├── 📁 raw/                 # Raw synthetic datasets
│   └── 📁 processed/           # Cleaned and transformed data
├── 📁 sql/                     # SQL analytics models
├── 📁 python/                  # Python analytics scripts
├── 📁 models/                  # ML models and predictions
├── 📁 dashboards/              # Dashboard templates and exports
├── 📁 notebooks/               # Jupyter notebooks for analysis
├── 📁 tests/                   # Unit tests
└── 📄 README.md
```

## 🚀 Core Features

### 1. **Synthetic Data Generation**
- Realistic customer journey simulation
- Multi-channel attribution tracking
- Betting behavior patterns
- Promotional campaign responses

### 2. **Marketing KPI Analytics**
- Customer Acquisition Cost (CAC)
- Cost Per Install (CPI)
- Return on Ad Spend (ROAS)
- Lifetime Value (LTV) modeling
- Cohort retention analysis

### 3. **A/B Testing Framework**
- Campaign effectiveness measurement
- Statistical significance testing
- Conversion uplift analysis
- Confidence interval calculation

### 4. **Predictive Models**
- Churn probability prediction
- LTV forecasting
- Conversion likelihood modeling
- Customer segmentation

### 5. **Interactive Dashboards**
- Real-time marketing metrics
- Campaign performance tracking
- Customer behavior insights
- Attribution model visualization

## 📊 Key Components

| Component | Description | Tools |
|-----------|-------------|-------|
| Data Pipeline | Synthetic data generation and processing | Python, Pandas, Faker |
| SQL Analytics | Marketing KPI calculations and transformations | PostgreSQL, DuckDB |
| ML Models | Predictive analytics for churn and LTV | Scikit-learn, XGBoost |
| Dashboards | Interactive visualizations | Streamlit, Power BI, Tableau |
| A/B Testing | Campaign measurement and experimentation | SciPy, Statsmodels |

## 🎮 Getting Started

### Prerequisites
```bash
Python 3.8+
PostgreSQL or DuckDB
Git
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd sportsbook-marketing-system

# Install Python dependencies
pip install -r requirements.txt

# Set up database
python python/setup_database.py

# Generate synthetic data
python python/data_generator.py
```

### Quick Start
```bash
# Generate sample dataset
python python/data_generator.py --customers 10000 --days 365

# Run SQL analytics
python python/run_analytics.py

# Launch dashboard
streamlit run dashboards/streamlit_dashboard.py
```

## 📈 Sample Metrics & KPIs

### Customer Acquisition
- **CPI (Cost Per Install)**: $15-45 depending on channel
- **CPA (Cost Per Acquisition)**: $75-150 for first deposit
- **Channel Performance**: Paid Search (35%), Affiliates (28%), Social (20%), Organic (17%)

### Customer Engagement
- **Average Session Length**: 18 minutes
- **Bets Per Session**: 3.2
- **Monthly Active Users**: 75% of registered users
- **Average Bet Size**: $25-85 depending on user segment

### Retention & Value
- **30-Day Retention**: 65%
- **90-Day Retention**: 35%
- **Customer LTV**: $450 (casual) to $2,300 (high-value)
- **Churn Rate**: 12% monthly

## 🔬 A/B Testing Examples

### Campaign: "Deposit $50, Get $50 Bonus"
- **Control Group**: Standard onboarding
- **Treatment Group**: Bonus offer
- **Metrics**: Conversion rate, deposit amount, 30-day retention
- **Expected Lift**: 15-25% conversion improvement

## 🎯 Business Applications

1. **Marketing Budget Optimization**: Allocate spend across channels based on ROAS
2. **Customer Segmentation**: Target high-value users with personalized campaigns
3. **Retention Programs**: Identify at-risk customers for re-engagement
4. **Product Development**: Understand user preferences and behavior patterns
5. **Competitive Analysis**: Benchmark performance against industry standards

## 📱 Dashboard Features

### Executive Summary
- Revenue and customer acquisition trends
- Channel performance overview
- Key metric health indicators

### Marketing Performance
- Campaign ROI analysis
- Attribution model comparison
- Customer acquisition funnel

### Customer Analytics
- Cohort retention curves
- Customer lifetime value trends
- Behavioral segmentation

### A/B Test Results
- Campaign performance comparison
- Statistical significance indicators
- Confidence interval visualization

## 🔧 Technical Stack

- **Data Processing**: Python, Pandas, NumPy
- **Database**: PostgreSQL, DuckDB
- **Analytics**: SQL, dbt (optional)
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Streamlit, Plotly, Seaborn
- **Dashboard**: Power BI, Tableau
- **Testing**: Pytest, SciPy
- **Version Control**: Git

## 📚 Learning Outcomes

By building this system, you'll master:
- End-to-end marketing analytics pipeline
- Customer lifecycle modeling
- A/B testing and experimentation
- Predictive analytics and ML
- Dashboard design and data visualization
- SQL analytics and data transformation
- Marketing attribution modeling

## 🎯 Next Steps

1. Generate synthetic data
2. Build SQL analytics models
3. Implement A/B testing framework
4. Create predictive models
5. Design interactive dashboards
6. Add real-time capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues, feature requests, and pull requests to improve this marketing intelligence system!