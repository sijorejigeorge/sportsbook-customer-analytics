# Tableau Dashboard Templates for Sportsbook Marketing Analytics

This directory contains Tableau dashboard templates (.twbx files) and documentation for creating comprehensive marketing analytics dashboards.

## 📊 Dashboard Templates

### 1. Executive Marketing Scorecard
**File**: `Executive_Marketing_Scorecard.twbx`

**Key Components**:
- High-level KPI summary
- Revenue and customer acquisition trends
- Channel performance overview
- Executive summary insights

**Key Calculated Fields**:
```tableau
// Customer Acquisition Cost
CAC: SUM([Acquisition Cost]) / COUNTD([Customer Id])

// Conversion Rate
Conversion Rate: 
COUNTD(IF [Is First Deposit] = TRUE THEN [Customer Id] END) / 
COUNTD([Customer Id])

// Customer Lifetime Value
Customer LTV: 
{ FIXED [Customer Id]: 
  SUM([Bet Amount] - [Payout]) - MAX([Acquisition Cost])
}

// ROAS (Return on Ad Spend)
ROAS: [Customer LTV] / [CAC]

// Days Since Signup
Days Since Signup: DATEDIFF('day', [Signup Date], TODAY())

// Customer Segment
Customer Segment:
IF [Total Wagered] >= 5000 THEN "High Value"
ELSEIF [Total Wagered] >= 1000 THEN "Medium Value"
ELSEIF [Total Wagered] >= 100 THEN "Low Value"
ELSE "Inactive"
END
```

### 2. Campaign Performance Analytics
**File**: `Campaign_Performance_Analytics.twbx`

**Key Components**:
- Campaign ROI analysis
- A/B test results
- Response rate tracking
- Attribution modeling

**Key Calculated Fields**:
```tableau
// Campaign ROI
Campaign ROI:
( [Campaign Revenue] - [Campaign Cost] ) / [Campaign Cost]

// Response Rate
Response Rate:
SUM(IF [Responded] = TRUE THEN 1 ELSE 0 END) / COUNT([Customer Id])

// Statistical Significance
Z Score:
( [Treatment Rate] - [Control Rate] ) / 
SQRT(
  [Pooled Rate] * (1 - [Pooled Rate]) * 
  ((1/[Treatment Size]) + (1/[Control Size]))
)

// Confidence Interval Lower
CI Lower:
[Treatment Rate] - 1.96 * 
SQRT([Treatment Rate] * (1-[Treatment Rate]) / [Treatment Size])

// Campaign Lift
Campaign Lift:
([Treatment Conversion] - [Control Conversion]) / [Control Conversion]

// Cost Per Response
CPR: [Campaign Cost] / [Total Responses]
```

### 3. Customer Journey & Cohort Analysis
**File**: `Customer_Journey_Cohorts.twbx`

**Key Components**:
- Cohort retention heatmaps
- Customer lifecycle stages
- Journey flow visualization
- Behavioral segmentation

**Key Calculated Fields**:
```tableau
// Cohort Period
Cohort Period: 
DATE(YEAR([Signup Date]), MONTH([Signup Date]), 1)

// Period Number
Period Number:
DATEDIFF('month', [Cohort Period], [Activity Period])

// Retention Rate
Retention Rate:
COUNTD([Customer Id]) / 
{FIXED [Cohort Period]: COUNTD([Customer Id])}

// RFM Scores
Recency Score:
IF [Days Since Last Activity] <= 7 THEN 5
ELSEIF [Days Since Last Activity] <= 30 THEN 4
ELSEIF [Days Since Last Activity] <= 90 THEN 3
ELSEIF [Days Since Last Activity] <= 180 THEN 2
ELSE 1
END

Frequency Score:
IF [Total Bets] >= 100 THEN 5
ELSEIF [Total Bets] >= 50 THEN 4
ELSEIF [Total Bets] >= 20 THEN 3
ELSEIF [Total Bets] >= 5 THEN 2
ELSEIF [Total Bets] >= 1 THEN 1
ELSE 0
END

Monetary Score:
IF [Total Wagered] >= 5000 THEN 5
ELSEIF [Total Wagered] >= 1000 THEN 4
ELSEIF [Total Wagered] >= 500 THEN 3
ELSEIF [Total Wagered] >= 100 THEN 2
ELSEIF [Total Wagered] >= 10 THEN 1
ELSE 0
END

// Customer Lifecycle Stage
Lifecycle Stage:
IF [Has Made Deposit] = FALSE THEN "Prospect"
ELSEIF [Days Since Last Activity] <= 30 THEN "Active"
ELSEIF [Days Since Last Activity] <= 90 THEN "At Risk"
ELSE "Churned"
END
```

### 4. Attribution & Channel Analysis
**File**: `Attribution_Channel_Analysis.twbx`

**Key Components**:
- Multi-touch attribution comparison
- Channel contribution analysis
- Customer journey mapping
- Touch point effectiveness

**Key Calculated Fields**:
```tableau
// Last Touch Attribution
Last Touch Attribution:
{FIXED [Customer Id]: 
  IF [Campaign Date] = {FIXED [Customer Id]: MAX([Campaign Date])}
  THEN [Conversion Value]
  ELSE 0
  END
}

// First Touch Attribution
First Touch Attribution:
{FIXED [Customer Id]: 
  IF [Campaign Date] = {FIXED [Customer Id]: MIN([Campaign Date])}
  THEN [Conversion Value]
  ELSE 0
  END
}

// Linear Attribution Weight
Linear Attribution Weight:
1 / {FIXED [Customer Id]: COUNTD([Campaign Id])}

// Time Decay Weight
Time Decay Weight:
POWER(0.5, DATEDIFF('day', [Campaign Date], [Conversion Date]) / 7)

// Position Based Attribution
Position Based Weight:
{FIXED [Customer Id]: 
  IF [Campaign Date] = MAX([Campaign Date]) OR 
     [Campaign Date] = MIN([Campaign Date]) 
  THEN 0.4
  ELSE 0.2 / (COUNTD([Campaign Id]) - 2)
  END
}

// Channel Efficiency Score
Channel Efficiency:
[Attributed Conversions] / [Channel Investment]
```

### 5. Real-Time Performance Monitor
**File**: `Real_Time_Performance_Monitor.twbx`

**Key Components**:
- Real-time KPI monitoring
- Live campaign performance
- Daily/hourly trends
- Alert indicators

**Key Calculated Fields**:
```tableau
// Today's Metrics
Todays Revenue:
IF DATEPART('day', [Transaction Date]) = DATEPART('day', TODAY())
THEN [Bet Amount] - [Payout]
ELSE 0
END

// Hour of Day Analysis
Hour of Day: DATEPART('hour', [Session Start])

// Daily Change
Daily Revenue Change:
([Todays Revenue] - [Yesterdays Revenue]) / [Yesterdays Revenue]

// Performance Alert
Performance Alert:
IF [Daily Revenue Change] < -0.10 THEN "🔴 Down >10%"
ELSEIF [Daily Revenue Change] > 0.10 THEN "🟢 Up >10%"
ELSE "🟡 Stable"
END

// Moving Average (7-day)
7 Day Moving Average:
WINDOW_AVG(SUM([Revenue]), -6, 0)

// Target Achievement
Target Achievement:
SUM([Revenue]) / SUM([Target Revenue])
```

## 🔧 Dashboard Setup Guide

### 1. Data Connection
1. Open Tableau Desktop
2. Connect to Data Source:
   - **Option A**: CSV files from `data/raw/`
   - **Option B**: Database connection (PostgreSQL/MySQL)
   - **Option C**: Excel workbook with all sheets

### 2. Data Source Configuration

**Join Configuration**:
```
Customers (Primary)
├── LEFT JOIN Transactions ON customers.customer_id = transactions.customer_id
├── LEFT JOIN BettingActivity ON customers.customer_id = betting_activity.customer_id
├── LEFT JOIN Sessions ON customers.customer_id = sessions.customer_id
└── LEFT JOIN CustomerCampaigns ON customers.customer_id = customer_campaigns.customer_id
    └── LEFT JOIN Campaigns ON customer_campaigns.campaign_id = campaigns.campaign_id
```

### 3. Data Preparation

**Custom Date Calculations**:
```tableau
// Fiscal Year (assuming April start)
Fiscal Year: 
IF MONTH([Date]) >= 4 
THEN YEAR([Date]) 
ELSE YEAR([Date]) - 1 
END

// Week Starting Monday
Week Start: DATEADD('day', 1-DATEPART('weekday', [Date]), [Date])

// Previous Period Calculations
Previous Month Revenue: 
{FIXED : 
  SUM(IF DATEPART('month', [Transaction Date]) = 
         DATEPART('month', TODAY()) - 1 
      THEN [Revenue] END)
}
```

### 4. Parameter Creation

**Date Range Parameter**:
- Name: Date Range Selection
- Data Type: String
- List: Last 7 Days, Last 30 Days, Last Quarter, Last Year

**Channel Filter Parameter**:
- Name: Channel Selection
- Data Type: String
- Allowable Values: List from [Acquisition Channel]

**Segment Parameter**:
- Name: Customer Segment
- Data Type: String
- List: All, High Value, Medium Value, Low Value

### 5. Dashboard Actions

**Filter Actions**:
```tableau
// Cross-dashboard filtering
Source Sheet: Channel Performance
Target: All sheets in dashboard
Fields: Acquisition Channel -> Acquisition Channel
```

**URL Actions**:
```tableau
// Deep dive into customer details
URL: https://customerdetails.com/customer/<Customer Id>
Target: New Tab
```

**Highlight Actions**:
```tableau
// Highlight related data points
Source: Cohort Heatmap
Target: Retention Trends
Fields: Cohort Period
```

## 📊 Visualization Best Practices

### 1. KPI Scorecards
- Use bullet charts for target vs actual
- Implement color coding for performance status
- Add sparklines for trend indication
- Include comparison to previous period

### 2. Time Series Analysis
- Use dual-axis for multiple metrics
- Add reference lines for targets/benchmarks
- Implement forecasting for future trends
- Color code based on performance thresholds

### 3. Cohort Heatmaps
- Use sequential color palette
- Add text annotations for key values
- Implement drill-down capabilities
- Include row and column totals

### 4. Attribution Analysis
- Use stacked bar charts for model comparison
- Implement waterfall charts for incremental contribution
- Add tooltips with detailed breakdowns
- Use consistent color coding across models

## 🎨 Design Guidelines

### Color Palette
```tableau
// Primary Colors
Brand Blue: #1f77b4
Success Green: #2ca02c
Warning Orange: #ff7f0e
Danger Red: #d62728
Neutral Gray: #7f7f7f

// Secondary Colors
Light Blue: #aec7e8
Light Green: #98df8a
Light Orange: #ffbb78
Light Red: #ff9896
Light Gray: #c5c5c5
```

### Typography
- **Headers**: Tableau Bold, 14-16pt
- **Subheaders**: Tableau Medium, 12pt
- **Body Text**: Tableau Regular, 10pt
- **KPI Values**: Tableau Bold, 18-24pt

### Layout Standards
- **Dashboard Size**: 1200x800 (Desktop), 800x1200 (Mobile)
- **Margin**: 10px consistent padding
- **Spacing**: 5px between related elements, 15px between sections
- **Grid**: 12-column responsive grid system

## 📱 Mobile Optimization

### Device-Specific Layouts
```tableau
// Mobile Layout Adjustments
- Single column layout
- Larger touch targets (minimum 44px)
- Simplified visualizations
- Scrollable content areas
- Collapsible sections
```

### Responsive Design
- Use device preview to test layouts
- Implement floating legends
- Adjust font sizes automatically
- Hide non-essential elements on small screens

## 🔄 Performance Optimization

### Extract Optimization
```tableau
// Extract Filters
- Date range: Last 2 years only
- Active customers only
- Exclude test accounts

// Extract Aggregations
- Pre-aggregate daily metrics
- Create materialized calculations
- Remove unused fields
```

### Dashboard Performance
- Limit number of marks (<50K per sheet)
- Use context filters appropriately
- Implement efficient joins
- Cache workbooks for better performance

## 📈 Advanced Analytics

### Statistical Functions
```tableau
// Correlation Analysis
Correlation: CORR([Metric1], [Metric2])

// Linear Regression
Trend Line: TREND([Revenue], [Date])

// Standard Deviation
Std Dev: STDEV([Revenue])

// Percentile Analysis
P95 Revenue: PERCENTILE([Revenue], 0.95)
```

### Forecasting
```tableau
// Forecast Parameters
- Model: Automatic
- Forecast Length: 3 months
- Prediction Intervals: 95%
- Seasonality: Automatic detection

// Custom Forecast
Custom Forecast: 
[Historical Average] * (1 + [Growth Rate])^([Future Periods])
```

## 🔧 Maintenance & Updates

### Data Refresh Schedule
- **Real-time**: Live connection (if database supports)
- **Daily**: Extract refresh at 6 AM
- **Weekly**: Full data refresh every Sunday
- **Monthly**: Historical data archive

### Version Control
- Use Tableau Server/Cloud for centralized management
- Implement development → staging → production workflow
- Document all changes in version notes
- Backup workbooks before major updates

### User Training
- Create user guides for each dashboard
- Conduct regular training sessions
- Implement feedback collection system
- Monitor usage analytics

## 📞 Support Resources

- **Tableau Documentation**: https://help.tableau.com/
- **Community Forums**: https://community.tableau.com/
- **Training Videos**: Tableau Training Library
- **Best Practices**: Tableau Zen Masters Blog

## 🎯 Dashboard Usage Guidelines

### Executive Dashboard
- **Audience**: C-level executives, VP Marketing
- **Update Frequency**: Daily
- **Key Questions**: Overall performance, ROI, strategic insights

### Campaign Dashboard
- **Audience**: Marketing managers, campaign analysts
- **Update Frequency**: Real-time during campaigns
- **Key Questions**: Campaign effectiveness, optimization opportunities

### Customer Analytics
- **Audience**: Customer success, retention teams
- **Update Frequency**: Weekly
- **Key Questions**: Churn risk, segment performance, lifecycle analysis

### Attribution Analysis
- **Audience**: Marketing analysts, media planners
- **Update Frequency**: Monthly
- **Key Questions**: Channel effectiveness, budget allocation, customer journey insights