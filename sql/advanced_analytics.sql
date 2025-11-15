-- =============================================
-- Advanced Analytics Queries
-- =============================================
-- This file contains sophisticated analytical queries for
-- advanced marketing intelligence and customer insights

-- Customer Journey Analysis
-- =============================================
CREATE VIEW v_customer_journey AS
WITH customer_milestones AS (
    SELECT 
        c.customer_id,
        c.signup_date,
        c.acquisition_channel,
        c.customer_segment,
        
        -- First deposit milestone
        MIN(CASE WHEN t.is_first_deposit THEN t.transaction_date END) as first_deposit_date,
        MIN(CASE WHEN t.is_first_deposit THEN t.amount END) as first_deposit_amount,
        
        -- First bet milestone
        MIN(ba.bet_date) as first_bet_date,
        MIN(CASE WHEN ba.bet_date = MIN(ba.bet_date) OVER (PARTITION BY ba.customer_id) 
                 THEN ba.bet_amount END) as first_bet_amount,
        
        -- Activity patterns
        COUNT(DISTINCT DATE(ba.bet_date)) as active_betting_days,
        COUNT(DISTINCT DATE(t.transaction_date)) as active_transaction_days,
        
        -- Last activity
        MAX(GREATEST(ba.bet_date, t.transaction_date)) as last_activity_date
        
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN betting_activity ba ON c.customer_id = ba.customer_id
    GROUP BY c.customer_id, c.signup_date, c.acquisition_channel, c.customer_segment
)
SELECT 
    *,
    -- Time to conversion metrics
    CASE WHEN first_deposit_date IS NOT NULL 
         THEN DATE_PART('day', first_deposit_date - signup_date)
         ELSE NULL END as days_to_first_deposit,
    
    CASE WHEN first_bet_date IS NOT NULL 
         THEN DATE_PART('day', first_bet_date - signup_date)
         ELSE NULL END as days_to_first_bet,
    
    -- Customer lifecycle stage
    CASE 
        WHEN first_deposit_date IS NULL THEN 'Registered - No Deposit'
        WHEN first_bet_date IS NULL THEN 'Deposited - No Bet'
        WHEN DATE_PART('day', CURRENT_DATE - last_activity_date) <= 30 THEN 'Active'
        WHEN DATE_PART('day', CURRENT_DATE - last_activity_date) <= 90 THEN 'Dormant'
        ELSE 'Churned'
    END as lifecycle_stage,
    
    -- Engagement intensity
    CASE 
        WHEN active_betting_days >= 30 THEN 'High Frequency'
        WHEN active_betting_days >= 10 THEN 'Medium Frequency'
        WHEN active_betting_days >= 1 THEN 'Low Frequency'
        ELSE 'No Betting'
    END as engagement_level
    
FROM customer_milestones;

-- Churn Prediction Dataset
-- =============================================
CREATE VIEW v_churn_features AS
WITH customer_activity AS (
    SELECT 
        c.customer_id,
        c.signup_date,
        c.acquisition_channel,
        c.customer_segment,
        c.age,
        
        -- Recency features (last 30 days)
        COUNT(CASE WHEN ba.bet_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as bets_last_30d,
        COALESCE(SUM(CASE WHEN ba.bet_date >= CURRENT_DATE - INTERVAL '30 days' THEN ba.bet_amount END), 0) as wagered_last_30d,
        COUNT(CASE WHEN t.transaction_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as transactions_last_30d,
        
        -- Frequency features (lifetime)
        COUNT(ba.bet_id) as total_bets,
        COUNT(DISTINCT DATE(ba.bet_date)) as betting_days,
        COUNT(DISTINCT s.session_id) as total_sessions,
        
        -- Monetary features
        COALESCE(SUM(ba.bet_amount), 0) as total_wagered,
        COALESCE(SUM(CASE WHEN t.transaction_type = 'deposit' THEN t.amount END), 0) as total_deposits,
        COALESCE(AVG(ba.bet_amount), 0) as avg_bet_size,
        
        -- Behavioral features
        COALESCE(AVG(s.duration_minutes), 0) as avg_session_duration,
        COUNT(DISTINCT ba.sport) as sports_diversity,
        STDDEV(ba.bet_amount) as bet_size_variance,
        
        -- Engagement consistency (coefficient of variation of daily betting)
        CASE WHEN COUNT(DISTINCT DATE(ba.bet_date)) > 1 THEN
            STDDEV(daily_bets.bets_per_day) / NULLIF(AVG(daily_bets.bets_per_day), 0)
        ELSE 0 END as betting_consistency,
        
        -- Last activity
        MAX(GREATEST(ba.bet_date, t.transaction_date)) as last_activity_date
        
    FROM customers c
    LEFT JOIN betting_activity ba ON c.customer_id = ba.customer_id
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN sessions s ON c.customer_id = s.customer_id
    LEFT JOIN (
        SELECT 
            customer_id, 
            DATE(bet_date) as bet_date,
            COUNT(*) as bets_per_day
        FROM betting_activity 
        GROUP BY customer_id, DATE(bet_date)
    ) daily_bets ON c.customer_id = daily_bets.customer_id
    GROUP BY c.customer_id, c.signup_date, c.acquisition_channel, c.customer_segment, c.age
)
SELECT 
    *,
    -- Target variable (churned if no activity in last 30 days and had prior activity)
    CASE 
        WHEN last_activity_date IS NULL THEN NULL  -- Never active
        WHEN DATE_PART('day', CURRENT_DATE - last_activity_date) > 30 AND total_bets > 0 THEN 1
        ELSE 0
    END as is_churned,
    
    -- Days since last activity
    CASE WHEN last_activity_date IS NOT NULL 
         THEN DATE_PART('day', CURRENT_DATE - last_activity_date)
         ELSE NULL END as days_since_last_activity,
    
    -- Customer tenure
    DATE_PART('day', CURRENT_DATE - signup_date) as tenure_days,
    
    -- Derived ratios
    CASE WHEN total_bets > 0 THEN bets_last_30d::DECIMAL / total_bets ELSE 0 END as recent_activity_ratio,
    CASE WHEN total_wagered > 0 THEN wagered_last_30d / total_wagered ELSE 0 END as recent_spending_ratio,
    CASE WHEN betting_days > 0 THEN total_bets::DECIMAL / betting_days ELSE 0 END as bets_per_active_day
    
FROM customer_activity;

-- Campaign Effectiveness Deep Dive
-- =============================================
CREATE VIEW v_campaign_deep_analysis AS
WITH campaign_customer_behavior AS (
    SELECT 
        cc.campaign_id,
        cc.customer_id,
        cc.responded,
        cc.bonus_claimed,
        c.campaign_type,
        c.start_date,
        c.end_date,
        cust.customer_segment,
        cust.acquisition_channel,
        
        -- Pre-campaign behavior (30 days before)
        COUNT(CASE WHEN ba.bet_date BETWEEN c.start_date - INTERVAL '30 days' AND c.start_date 
                   THEN 1 END) as pre_campaign_bets,
        COALESCE(SUM(CASE WHEN ba.bet_date BETWEEN c.start_date - INTERVAL '30 days' AND c.start_date 
                          THEN ba.bet_amount END), 0) as pre_campaign_wagered,
        
        -- During campaign behavior
        COUNT(CASE WHEN ba.bet_date BETWEEN c.start_date AND c.end_date 
                   THEN 1 END) as during_campaign_bets,
        COALESCE(SUM(CASE WHEN ba.bet_date BETWEEN c.start_date AND c.end_date 
                          THEN ba.bet_amount END), 0) as during_campaign_wagered,
        
        -- Post-campaign behavior (30 days after)
        COUNT(CASE WHEN ba.bet_date BETWEEN c.end_date AND c.end_date + INTERVAL '30 days' 
                   THEN 1 END) as post_campaign_bets,
        COALESCE(SUM(CASE WHEN ba.bet_date BETWEEN c.end_date AND c.end_date + INTERVAL '30 days' 
                          THEN ba.bet_amount END), 0) as post_campaign_wagered,
        
        -- Deposits during campaign period
        COALESCE(SUM(CASE WHEN t.transaction_type = 'deposit' 
                          AND t.transaction_date BETWEEN c.start_date AND c.end_date + INTERVAL '7 days'
                          THEN t.amount END), 0) as campaign_period_deposits
        
    FROM customer_campaigns cc
    JOIN campaigns c ON cc.campaign_id = c.campaign_id
    JOIN customers cust ON cc.customer_id = cust.customer_id
    LEFT JOIN betting_activity ba ON cc.customer_id = ba.customer_id
    LEFT JOIN transactions t ON cc.customer_id = t.customer_id
    GROUP BY cc.campaign_id, cc.customer_id, cc.responded, cc.bonus_claimed, 
             c.campaign_type, c.start_date, c.end_date, cust.customer_segment, cust.acquisition_channel
)
SELECT 
    campaign_id,
    campaign_type,
    customer_segment,
    acquisition_channel,
    
    -- Response analysis
    COUNT(*) as customers_targeted,
    COUNT(CASE WHEN responded THEN 1 END) as customers_responded,
    COUNT(CASE WHEN bonus_claimed THEN 1 END) as customers_claimed_bonus,
    
    ROUND(COUNT(CASE WHEN responded THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as response_rate,
    ROUND(COUNT(CASE WHEN bonus_claimed THEN 1 END)::DECIMAL / NULLIF(COUNT(CASE WHEN responded THEN 1 END), 0) * 100, 2) as claim_rate,
    
    -- Behavior change analysis (responders vs non-responders)
    ROUND(AVG(CASE WHEN responded THEN during_campaign_wagered END), 2) as avg_wagered_responders,
    ROUND(AVG(CASE WHEN NOT responded THEN during_campaign_wagered END), 2) as avg_wagered_non_responders,
    
    -- Lift analysis
    ROUND(AVG(CASE WHEN responded THEN during_campaign_wagered - pre_campaign_wagered END), 2) as avg_wagered_lift_responders,
    ROUND(AVG(CASE WHEN NOT responded THEN during_campaign_wagered - pre_campaign_wagered END), 2) as avg_wagered_lift_non_responders,
    
    -- Revenue impact
    SUM(campaign_period_deposits) as total_deposits_generated,
    ROUND(AVG(campaign_period_deposits), 2) as avg_deposit_per_customer,
    
    -- Post-campaign retention
    ROUND(COUNT(CASE WHEN responded AND post_campaign_bets > 0 THEN 1 END)::DECIMAL / 
          NULLIF(COUNT(CASE WHEN responded THEN 1 END), 0) * 100, 2) as responder_retention_rate,
    ROUND(COUNT(CASE WHEN NOT responded AND post_campaign_bets > 0 THEN 1 END)::DECIMAL / 
          NULLIF(COUNT(CASE WHEN NOT responded THEN 1 END), 0) * 100, 2) as non_responder_retention_rate
    
FROM campaign_customer_behavior
GROUP BY campaign_id, campaign_type, customer_segment, acquisition_channel
ORDER BY campaign_id, customer_segment;

-- Sports Preference Analysis
-- =============================================
CREATE VIEW v_sports_preference_analysis AS
WITH customer_sport_behavior AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.acquisition_channel,
        ba.sport,
        COUNT(ba.bet_id) as bets_count,
        SUM(ba.bet_amount) as total_wagered,
        AVG(ba.bet_amount) as avg_bet_size,
        SUM(CASE WHEN ba.bet_won THEN 1 ELSE 0 END)::DECIMAL / COUNT(ba.bet_id) as win_rate,
        SUM(ba.net_result) as net_result
        
    FROM customers c
    JOIN betting_activity ba ON c.customer_id = ba.customer_id
    GROUP BY c.customer_id, c.customer_segment, c.acquisition_channel, ba.sport
),
customer_sport_preferences AS (
    SELECT 
        customer_id,
        customer_segment,
        acquisition_channel,
        sport,
        bets_count,
        total_wagered,
        avg_bet_size,
        win_rate,
        net_result,
        
        -- Rank sports by betting volume for each customer
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY total_wagered DESC) as sport_rank_by_volume,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY bets_count DESC) as sport_rank_by_frequency
        
    FROM customer_sport_behavior
)
SELECT 
    sport,
    customer_segment,
    acquisition_channel,
    
    -- Overall metrics
    COUNT(DISTINCT customer_id) as customers_betting_sport,
    SUM(bets_count) as total_bets,
    SUM(total_wagered) as total_volume,
    ROUND(AVG(avg_bet_size), 2) as avg_bet_size,
    ROUND(AVG(win_rate) * 100, 2) as avg_win_rate_pct,
    
    -- Preference metrics
    COUNT(CASE WHEN sport_rank_by_volume = 1 THEN 1 END) as customers_primary_sport_by_volume,
    COUNT(CASE WHEN sport_rank_by_frequency = 1 THEN 1 END) as customers_primary_sport_by_frequency,
    
    -- Profitability
    SUM(net_result) as total_customer_net_result,
    -SUM(net_result) as house_edge_revenue,  -- Negative of customer result is house revenue
    ROUND(-SUM(net_result) / SUM(total_wagered) * 100, 2) as house_edge_percentage
    
FROM customer_sport_preferences
GROUP BY sport, customer_segment, acquisition_channel
ORDER BY total_volume DESC;

-- Time-Series Analysis for Forecasting
-- =============================================
CREATE VIEW v_daily_metrics_timeseries AS
WITH daily_aggregates AS (
    SELECT 
        DATE(activity_date) as date,
        daily_active_users,
        daily_betting_users,
        total_bets,
        total_wagered,
        gross_gaming_revenue,
        deposits_amount,
        avg_bet_size,
        
        -- Calculate moving averages (7-day)
        AVG(daily_active_users) OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma7_active_users,
        AVG(total_wagered) OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma7_wagered,
        AVG(gross_gaming_revenue) OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma7_ggr,
        
        -- Calculate day-over-day changes
        LAG(daily_active_users) OVER (ORDER BY activity_date) as prev_day_active_users,
        LAG(total_wagered) OVER (ORDER BY activity_date) as prev_day_wagered,
        LAG(gross_gaming_revenue) OVER (ORDER BY activity_date) as prev_day_ggr
        
    FROM v_daily_engagement
    ORDER BY activity_date
)
SELECT 
    date,
    daily_active_users,
    daily_betting_users,
    total_bets,
    total_wagered,
    gross_gaming_revenue,
    deposits_amount,
    avg_bet_size,
    
    -- Moving averages
    ROUND(ma7_active_users, 0) as ma7_active_users,
    ROUND(ma7_wagered, 2) as ma7_wagered,
    ROUND(ma7_ggr, 2) as ma7_ggr,
    
    -- Day-over-day percent changes
    ROUND((daily_active_users - prev_day_active_users)::DECIMAL / NULLIF(prev_day_active_users, 0) * 100, 2) as dod_change_active_users,
    ROUND((total_wagered - prev_day_wagered) / NULLIF(prev_day_wagered, 0) * 100, 2) as dod_change_wagered,
    ROUND((gross_gaming_revenue - prev_day_ggr) / NULLIF(prev_day_ggr, 0) * 100, 2) as dod_change_ggr,
    
    -- Day of week analysis
    EXTRACT(DOW FROM date) as day_of_week,  -- 0=Sunday, 6=Saturday
    CASE EXTRACT(DOW FROM date)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday' 
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_name,
    
    -- Seasonality indicators
    EXTRACT(QUARTER FROM date) as quarter,
    EXTRACT(MONTH FROM date) as month,
    EXTRACT(WEEK FROM date) as week_of_year
    
FROM daily_aggregates;

-- Advanced Cohort Analysis with Revenue
-- =============================================
CREATE VIEW v_revenue_cohort_analysis AS
WITH customer_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', signup_date) as cohort_month,
        signup_date
    FROM customers
),
customer_monthly_revenue AS (
    SELECT 
        cc.customer_id,
        cc.cohort_month,
        DATE_TRUNC('month', ba.bet_date) as activity_month,
        SUM(ba.bet_amount - ba.payout) as monthly_ggr,
        COUNT(DISTINCT DATE(ba.bet_date)) as active_days_in_month
    FROM customer_cohorts cc
    LEFT JOIN betting_activity ba ON cc.customer_id = ba.customer_id
    WHERE ba.bet_date IS NOT NULL
    GROUP BY cc.customer_id, cc.cohort_month, DATE_TRUNC('month', ba.bet_date)
),
cohort_revenue_summary AS (
    SELECT 
        cohort_month,
        activity_month,
        DATE_PART('month', AGE(activity_month, cohort_month)) as months_since_signup,
        
        COUNT(DISTINCT customer_id) as active_customers,
        SUM(monthly_ggr) as total_ggr,
        AVG(monthly_ggr) as avg_ggr_per_customer,
        SUM(active_days_in_month) as total_active_days,
        AVG(active_days_in_month) as avg_active_days_per_customer
        
    FROM customer_monthly_revenue
    GROUP BY cohort_month, activity_month
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM customer_cohorts
    GROUP BY cohort_month
)
SELECT 
    crs.cohort_month,
    cs.cohort_size,
    crs.months_since_signup,
    crs.active_customers,
    ROUND(crs.active_customers::DECIMAL / cs.cohort_size * 100, 2) as retention_rate,
    
    ROUND(crs.total_ggr, 2) as cohort_monthly_ggr,
    ROUND(crs.avg_ggr_per_customer, 2) as avg_ggr_per_active_customer,
    ROUND(crs.total_ggr / cs.cohort_size, 2) as ggr_per_cohort_customer,
    
    ROUND(crs.avg_active_days_per_customer, 1) as avg_active_days,
    
    -- Cumulative metrics
    SUM(crs.total_ggr) OVER (
        PARTITION BY crs.cohort_month 
        ORDER BY crs.months_since_signup 
        ROWS UNBOUNDED PRECEDING
    ) as cumulative_ggr_per_cohort,
    
    ROUND(
        SUM(crs.total_ggr) OVER (
            PARTITION BY crs.cohort_month 
            ORDER BY crs.months_since_signup 
            ROWS UNBOUNDED PRECEDING
        ) / cs.cohort_size, 2
    ) as cumulative_ggr_per_customer
    
FROM cohort_revenue_summary crs
JOIN cohort_sizes cs ON crs.cohort_month = cs.cohort_month
ORDER BY crs.cohort_month, crs.months_since_signup;