# Power BI Dashboard Templates for Sportsbook Marketing Analytics

This directory contains Power BI dashboard templates (.pbix files) and documentation for creating comprehensive marketing analytics dashboards.

## 📊 Dashboard Templates

### 1. Executive Marketing Overview
**File**: `Executive_Marketing_Overview.pbix`

**Key Components**:
- Revenue and customer acquisition trends
- Channel performance overview
- Customer lifetime value metrics
- Conversion funnel analysis

**Key Visualizations**:
- KPI cards for total customers, conversion rate, ROAS
- Time series charts for daily/monthly trends
- Bar charts for channel performance
- Funnel visualization for customer acquisition

**DAX Measures**:
```dax
// Customer Acquisition Cost
CAC = DIVIDE(SUM(Customers[acquisition_cost]), DISTINCTCOUNT(Customers[customer_id]))

// Conversion Rate
Conversion Rate = DIVIDE(
    DISTINCTCOUNT(Transactions[customer_id]),
    DISTINCTCOUNT(Customers[customer_id])
)

// Customer Lifetime Value
Customer LTV = 
VAR GrossRevenue = SUMX(
    BettingActivity,
    BettingActivity[bet_amount] - BettingActivity[payout]
)
VAR AcquisitionCost = SUM(Customers[acquisition_cost])
RETURN GrossRevenue - AcquisitionCost

// ROAS (Return on Ad Spend)
ROAS = DIVIDE([Customer LTV], [CAC])

// Monthly Active Users
MAU = DISTINCTCOUNT(BettingActivity[customer_id])

// Average Session Duration
Avg Session Duration = AVERAGE(Sessions[duration_minutes])
```

### 2. Campaign Performance Dashboard
**File**: `Campaign_Performance_Dashboard.pbix`

**Key Components**:
- Campaign ROI analysis
- A/B test results visualization
- Response rate tracking
- Budget allocation insights

**Key Visualizations**:
- Campaign performance matrix
- ROI waterfall charts
- Response rate trends
- Budget vs. results scatter plots

**DAX Measures**:
```dax
// Campaign ROI
Campaign ROI = 
VAR CampaignRevenue = 
    CALCULATE(
        [Customer LTV],
        CustomerCampaigns[responded] = TRUE
    )
VAR CampaignCost = SUM(Campaigns[cost_per_customer]) * COUNT(CustomerCampaigns[customer_id])
RETURN DIVIDE(CampaignRevenue - CampaignCost, CampaignCost)

// Response Rate
Response Rate = DIVIDE(
    COUNT(CustomerCampaigns[responded] = TRUE),
    COUNT(CustomerCampaigns[customer_id])
)

// Campaign Lift
Campaign Lift = 
VAR TreatmentConversion = 
    CALCULATE(
        [Conversion Rate],
        CustomerCampaigns[responded] = TRUE
    )
VAR ControlConversion = 
    CALCULATE(
        [Conversion Rate],
        CustomerCampaigns[responded] = FALSE
    )
RETURN DIVIDE(TreatmentConversion - ControlConversion, ControlConversion)
```

### 3. Customer Analytics Dashboard
**File**: `Customer_Analytics_Dashboard.pbix`

**Key Components**:
- Customer segmentation analysis
- Cohort retention curves
- Behavioral analytics
- Churn risk assessment

**Key Visualizations**:
- Customer segment pie charts
- Cohort retention heatmaps
- RFM analysis scatter plots
- Churn prediction gauges

**DAX Measures**:
```dax
// Customer Segment
Customer Segment = 
SWITCH(
    TRUE(),
    [Total Wagered] >= 5000, "High Value",
    [Total Wagered] >= 1000, "Medium Value",
    [Total Wagered] >= 100, "Low Value",
    "Inactive"
)

// Recency Score
Recency Score = 
VAR DaysSinceLastActivity = 
    DATEDIFF(
        MAX(BettingActivity[bet_date]),
        TODAY(),
        DAY
    )
RETURN
    SWITCH(
        TRUE(),
        DaysSinceLastActivity <= 7, 5,
        DaysSinceLastActivity <= 30, 4,
        DaysSinceLastActivity <= 90, 3,
        DaysSinceLastActivity <= 180, 2,
        1
    )

// Cohort Retention Rate
Cohort Retention = 
VAR CohortMonth = FORMAT(Customers[signup_date], "YYYY-MM")
VAR ActivityMonth = FORMAT(BettingActivity[bet_date], "YYYY-MM")
VAR CohortSize = 
    CALCULATE(
        DISTINCTCOUNT(Customers[customer_id]),
        ALL(BettingActivity)
    )
VAR ActiveInPeriod = DISTINCTCOUNT(BettingActivity[customer_id])
RETURN DIVIDE(ActiveInPeriod, CohortSize)
```

### 4. Attribution Analysis Dashboard
**File**: `Attribution_Analysis_Dashboard.pbix`

**Key Components**:
- Multi-touch attribution modeling
- Channel contribution analysis
- Customer journey visualization
- Attribution model comparison

**Key Visualizations**:
- Attribution model comparison charts
- Customer journey flow diagrams
- Channel contribution waterfall
- Touch point analysis tables

**DAX Measures**:
```dax
// Last-Touch Attribution
Last Touch Attribution = 
CALCULATE(
    [Conversion Rate],
    TOPN(1, CustomerCampaigns, CustomerCampaigns[sent_date], DESC)
)

// First-Touch Attribution
First Touch Attribution = 
CALCULATE(
    [Conversion Rate],
    TOPN(1, CustomerCampaigns, CustomerCampaigns[sent_date], ASC)
)

// Linear Attribution
Linear Attribution = 
VAR TouchpointCount = DISTINCTCOUNT(CustomerCampaigns[campaign_id])
RETURN DIVIDE([Conversion Rate], TouchpointCount)

// Time Decay Weight
Time Decay Weight = 
VAR DaysFromConversion = 
    DATEDIFF(
        CustomerCampaigns[sent_date],
        Transactions[transaction_date],
        DAY
    )
RETURN POWER(0.5, DaysFromConversion / 7)
```

## 🔧 Setup Instructions

### 1. Data Connection
1. Open Power BI Desktop
2. Get Data > Text/CSV
3. Connect to the CSV files in `data/raw/`:
   - customers.csv
   - transactions.csv
   - betting_activity.csv
   - sessions.csv
   - campaigns.csv
   - customer_campaigns.csv

### 2. Data Model Setup
1. Create relationships between tables:
   - Customers[customer_id] ↔ Transactions[customer_id]
   - Customers[customer_id] ↔ BettingActivity[customer_id]
   - Customers[customer_id] ↔ Sessions[customer_id]
   - Campaigns[campaign_id] ↔ CustomerCampaigns[campaign_id]
   - Customers[customer_id] ↔ CustomerCampaigns[customer_id]

2. Set up date tables for time intelligence:
   ```dax
   Date = CALENDAR(DATE(2024,1,1), DATE(2024,12,31))
   ```

3. Mark date columns as date type:
   - Customers[signup_date]
   - Transactions[transaction_date]
   - BettingActivity[bet_date]
   - Sessions[start_time]

### 3. Calculated Columns
```dax
// In Customers table
Customers[Tenure Days] = DATEDIFF(Customers[signup_date], TODAY(), DAY)

Customers[Has Deposited] = 
IF(
    RELATED(Transactions[is_first_deposit]) = TRUE,
    "Yes",
    "No"
)

// In BettingActivity table
BettingActivity[Gross Gaming Revenue] = 
BettingActivity[bet_amount] - BettingActivity[payout]

// In Sessions table
Sessions[Session Hour] = HOUR(Sessions[start_time])
```

### 4. Custom Visuals
Install these custom visuals from AppSource:
- **Hierarchy Slicer** - For filtering by date ranges
- **Sankey Chart** - For customer journey visualization
- **Waterfall Chart** - For attribution analysis
- **Calendar Visual** - For date-based analysis

### 5. Dashboard Themes
Apply consistent branding using custom themes:
```json
{
  "name": "Sportsbook Theme",
  "dataColors": [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
  ],
  "background": "#ffffff",
  "foreground": "#000000",
  "tableAccent": "#1f77b4"
}
```

## 📈 Key Metrics Reference

### Customer Metrics
- **Customer Acquisition Cost (CAC)**: Total acquisition spend / Number of new customers
- **Customer Lifetime Value (LTV)**: Total gross gaming revenue - Acquisition cost
- **LTV:CAC Ratio**: Customer LTV / Customer acquisition cost
- **Conversion Rate**: Customers who made first deposit / Total customers

### Engagement Metrics
- **Daily Active Users (DAU)**: Unique customers who placed bets on a given day
- **Monthly Active Users (MAU)**: Unique customers who placed bets in a given month
- **Session Duration**: Average time spent per session
- **Bet Frequency**: Average number of bets per active day

### Revenue Metrics
- **Gross Gaming Revenue (GGR)**: Total wagered - Total payouts
- **Net Gaming Revenue (NGR)**: GGR - Bonuses and promotional costs
- **Average Revenue Per User (ARPU)**: Total revenue / Total active users
- **Return on Ad Spend (ROAS)**: Revenue attributed to marketing / Marketing spend

### Campaign Metrics
- **Response Rate**: Customers who responded / Customers targeted
- **Conversion Lift**: (Treatment conversion rate - Control conversion rate) / Control conversion rate
- **Campaign ROI**: (Campaign revenue - Campaign cost) / Campaign cost
- **Cost Per Response (CPR)**: Total campaign cost / Number of responses

## 🎯 Best Practices

### 1. Performance Optimization
- Use DirectQuery for large datasets
- Implement incremental refresh for daily updates
- Create aggregation tables for improved performance
- Use variables in DAX for complex calculations

### 2. User Experience
- Implement drill-through pages for detailed analysis
- Use consistent color schemes and formatting
- Add tooltips with additional context
- Create mobile-optimized layouts

### 3. Security
- Implement Row-Level Security (RLS) for multi-tenant scenarios
- Use service principals for automated refresh
- Restrict sensitive data access based on user roles

### 4. Maintenance
- Document all custom measures and calculations
- Implement data quality checks
- Schedule regular data refresh
- Monitor dashboard usage and performance

## 📱 Mobile Optimization

Configure mobile layouts for key dashboards:
- Optimize KPI cards for mobile view
- Use stacked visualizations
- Implement touch-friendly navigation
- Test on various device sizes

## 🔄 Refresh Schedule

Set up automatic refresh schedule:
- **Real-time data**: Every 15 minutes (if using DirectQuery)
- **Daily aggregates**: Once per day at 6 AM
- **Monthly reports**: First day of each month
- **Campaign data**: After each campaign launch

## 📞 Support and Resources

- **Power BI Documentation**: https://docs.microsoft.com/en-us/power-bi/
- **DAX Reference**: https://docs.microsoft.com/en-us/dax/
- **Community Forums**: https://community.powerbi.com/
- **Training Materials**: Microsoft Learn Power BI path