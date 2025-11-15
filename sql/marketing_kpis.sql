-- =============================================
-- Sportsbook Marketing KPI Analytics
-- =============================================
-- This file contains SQL models for calculating key marketing metrics
-- including customer acquisition costs, lifetime value, cohort analysis,
-- and campaign performance measurement.

-- Customer Acquisition Cost (CAC) by Channel
-- =============================================
CREATE VIEW v_customer_acquisition_cost AS
WITH channel_metrics AS (
    SELECT 
        acquisition_channel,
        COUNT(DISTINCT customer_id) as total_customers,
        SUM(acquisition_cost) as total_acquisition_cost,
        COUNT(DISTINCT CASE WHEN customer_id IN (
            SELECT DISTINCT customer_id FROM transactions WHERE is_first_deposit = true
        ) THEN customer_id END) as converted_customers
    FROM customers
    GROUP BY acquisition_channel
)
SELECT 
    acquisition_channel,
    total_customers,
    converted_customers,
    ROUND(converted_customers::DECIMAL / total_customers * 100, 2) as conversion_rate_pct,
    ROUND(total_acquisition_cost / total_customers, 2) as cost_per_install_cpi,
    ROUND(total_acquisition_cost / NULLIF(converted_customers, 0), 2) as cost_per_acquisition_cpa,
    total_acquisition_cost
FROM channel_metrics
ORDER BY total_customers DESC;

-- Customer Lifetime Value (LTV) Analysis
-- =============================================
CREATE VIEW v_customer_ltv_analysis AS
WITH customer_financials AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.acquisition_channel,
        c.signup_date,
        c.acquisition_cost,
        
        -- Transaction metrics
        COALESCE(SUM(CASE WHEN t.transaction_type = 'deposit' THEN t.amount ELSE 0 END), 0) as total_deposits,
        COALESCE(SUM(CASE WHEN t.transaction_type = 'withdrawal' THEN t.amount ELSE 0 END), 0) as total_withdrawals,
        COALESCE(COUNT(CASE WHEN t.transaction_type = 'deposit' THEN 1 END), 0) as num_deposits,
        
        -- Betting metrics
        COALESCE(SUM(b.bet_amount), 0) as total_wagered,
        COALESCE(SUM(b.payout), 0) as total_payouts,
        COALESCE(SUM(b.net_result), 0) as customer_net_result,
        COALESCE(COUNT(b.bet_id), 0) as total_bets,
        
        -- Date ranges for tenure calculation
        MIN(t.transaction_date) as first_transaction_date,
        MAX(GREATEST(t.transaction_date, b.bet_date)) as last_activity_date
        
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN betting_activity b ON c.customer_id = b.customer_id
    GROUP BY c.customer_id, c.customer_segment, c.acquisition_channel, c.signup_date, c.acquisition_cost
),
ltv_calculations AS (
    SELECT 
        *,
        total_deposits - total_withdrawals as net_deposits,
        total_wagered - total_payouts as gross_gaming_revenue,
        CASE 
            WHEN last_activity_date IS NOT NULL 
            THEN DATE_PART('day', last_activity_date - signup_date) + 1
            ELSE 0 
        END as tenure_days,
        
        -- LTV calculation (gross gaming revenue - acquisition cost)
        (total_wagered - total_payouts) - acquisition_cost as customer_ltv
    FROM customer_financials
)
SELECT 
    customer_id,
    customer_segment,
    acquisition_channel,
    signup_date,
    acquisition_cost,
    total_deposits,
    total_withdrawals,
    net_deposits,
    total_wagered,
    total_payouts,
    gross_gaming_revenue,
    customer_ltv,
    tenure_days,
    total_bets,
    CASE 
        WHEN total_bets > 0 THEN ROUND(total_wagered / total_bets, 2)
        ELSE 0 
    END as avg_bet_size,
    CASE 
        WHEN tenure_days > 0 THEN ROUND(customer_ltv / (tenure_days / 30.0), 2)
        ELSE 0 
    END as monthly_ltv,
    first_transaction_date,
    last_activity_date
FROM ltv_calculations;

-- LTV Summary by Segment and Channel
-- =============================================
CREATE VIEW v_ltv_summary AS
SELECT 
    customer_segment,
    acquisition_channel,
    COUNT(*) as customer_count,
    ROUND(AVG(customer_ltv), 2) as avg_ltv,
    ROUND(MEDIAN(customer_ltv), 2) as median_ltv,
    ROUND(AVG(acquisition_cost), 2) as avg_acquisition_cost,
    ROUND(AVG(customer_ltv) / AVG(acquisition_cost), 2) as ltv_to_cac_ratio,
    ROUND(AVG(tenure_days), 1) as avg_tenure_days,
    ROUND(AVG(total_wagered), 2) as avg_total_wagered,
    ROUND(SUM(gross_gaming_revenue), 2) as total_ggr
FROM v_customer_ltv_analysis
WHERE total_bets > 0  -- Only include customers who actually bet
GROUP BY customer_segment, acquisition_channel
ORDER BY avg_ltv DESC;

-- Cohort Retention Analysis
-- =============================================
CREATE VIEW v_cohort_analysis AS
WITH customer_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', signup_date) as cohort_month,
        signup_date
    FROM customers
),
customer_activities AS (
    SELECT 
        cc.customer_id,
        cc.cohort_month,
        cc.signup_date,
        DATE_TRUNC('month', COALESCE(ba.bet_date, t.transaction_date)) as activity_month
    FROM customer_cohorts cc
    LEFT JOIN betting_activity ba ON cc.customer_id = ba.customer_id
    LEFT JOIN transactions t ON cc.customer_id = t.customer_id
    WHERE COALESCE(ba.bet_date, t.transaction_date) IS NOT NULL
),
cohort_data AS (
    SELECT 
        cohort_month,
        activity_month,
        DATE_PART('month', AGE(activity_month, cohort_month)) as months_since_signup,
        COUNT(DISTINCT customer_id) as active_customers
    FROM customer_activities
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
    cd.cohort_month,
    cs.cohort_size,
    cd.months_since_signup,
    cd.active_customers,
    ROUND(cd.active_customers::DECIMAL / cs.cohort_size * 100, 2) as retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.months_since_signup;

-- Campaign Performance Analysis
-- =============================================
CREATE VIEW v_campaign_performance AS
WITH campaign_metrics AS (
    SELECT 
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        c.start_date,
        c.end_date,
        c.cost_per_customer,
        
        COUNT(cc.customer_id) as customers_targeted,
        COUNT(CASE WHEN cc.responded = true THEN 1 END) as customers_responded,
        COUNT(CASE WHEN cc.bonus_claimed = true THEN 1 END) as bonuses_claimed,
        
        SUM(cc.campaign_cost) as total_campaign_cost
    FROM campaigns c
    LEFT JOIN customer_campaigns cc ON c.campaign_id = cc.campaign_id
    GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.start_date, c.end_date, c.cost_per_customer
),
campaign_revenue AS (
    SELECT 
        cc.campaign_id,
        
        -- Revenue from customers who responded to campaign (30 days post-campaign)
        SUM(CASE WHEN t.transaction_type = 'deposit' AND t.transaction_date BETWEEN c.start_date AND c.end_date + INTERVAL '30 days'
            THEN t.amount ELSE 0 END) as campaign_period_deposits,
            
        SUM(CASE WHEN ba.bet_date BETWEEN c.start_date AND c.end_date + INTERVAL '30 days'
            THEN ba.bet_amount - ba.payout ELSE 0 END) as campaign_period_ggr
            
    FROM customer_campaigns cc
    JOIN campaigns c ON cc.campaign_id = c.campaign_id
    LEFT JOIN transactions t ON cc.customer_id = t.customer_id AND cc.responded = true
    LEFT JOIN betting_activity ba ON cc.customer_id = ba.customer_id AND cc.responded = true
    GROUP BY cc.campaign_id
)
SELECT 
    cm.campaign_id,
    cm.campaign_name,
    cm.campaign_type,
    cm.start_date,
    cm.end_date,
    cm.customers_targeted,
    cm.customers_responded,
    cm.bonuses_claimed,
    cm.total_campaign_cost,
    COALESCE(cr.campaign_period_deposits, 0) as revenue_deposits,
    COALESCE(cr.campaign_period_ggr, 0) as revenue_ggr,
    
    -- Performance metrics
    ROUND(cm.customers_responded::DECIMAL / NULLIF(cm.customers_targeted, 0) * 100, 2) as response_rate_pct,
    ROUND(cm.bonuses_claimed::DECIMAL / NULLIF(cm.customers_responded, 0) * 100, 2) as claim_rate_pct,
    ROUND(COALESCE(cr.campaign_period_ggr, 0) / NULLIF(cm.total_campaign_cost, 0), 2) as roas_ratio,
    ROUND(cm.total_campaign_cost / NULLIF(cm.customers_responded, 0), 2) as cost_per_response
    
FROM campaign_metrics cm
LEFT JOIN campaign_revenue cr ON cm.campaign_id = cr.campaign_id
ORDER BY cm.start_date DESC;

-- Daily Active Users and Engagement Metrics
-- =============================================
CREATE VIEW v_daily_engagement AS
WITH daily_activity AS (
    SELECT 
        DATE(COALESCE(ba.bet_date, t.transaction_date, s.start_time)) as activity_date,
        COUNT(DISTINCT COALESCE(ba.customer_id, t.customer_id, s.customer_id)) as daily_active_users,
        COUNT(DISTINCT ba.customer_id) as daily_betting_users,
        COUNT(DISTINCT t.customer_id) as daily_transacting_users,
        COUNT(DISTINCT s.customer_id) as daily_session_users,
        
        -- Betting metrics
        COUNT(ba.bet_id) as total_bets,
        COALESCE(SUM(ba.bet_amount), 0) as total_wagered,
        COALESCE(SUM(ba.payout), 0) as total_payouts,
        COALESCE(SUM(ba.bet_amount - ba.payout), 0) as gross_gaming_revenue,
        
        -- Transaction metrics
        COUNT(CASE WHEN t.transaction_type = 'deposit' THEN 1 END) as deposits_count,
        COALESCE(SUM(CASE WHEN t.transaction_type = 'deposit' THEN t.amount END), 0) as deposits_amount,
        
        -- Session metrics
        COUNT(s.session_id) as total_sessions,
        COALESCE(AVG(s.duration_minutes), 0) as avg_session_duration
        
    FROM customers c
    LEFT JOIN betting_activity ba ON c.customer_id = ba.customer_id
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN sessions s ON c.customer_id = s.customer_id
    WHERE COALESCE(ba.bet_date, t.transaction_date, s.start_time) IS NOT NULL
    GROUP BY DATE(COALESCE(ba.bet_date, t.transaction_date, s.start_time))
)
SELECT 
    activity_date,
    daily_active_users,
    daily_betting_users,
    daily_transacting_users,
    daily_session_users,
    total_bets,
    total_wagered,
    total_payouts,
    gross_gaming_revenue,
    deposits_count,
    deposits_amount,
    total_sessions,
    ROUND(avg_session_duration, 2) as avg_session_duration,
    
    -- Derived metrics
    CASE 
        WHEN daily_betting_users > 0 THEN ROUND(total_bets::DECIMAL / daily_betting_users, 2)
        ELSE 0 
    END as bets_per_betting_user,
    CASE 
        WHEN total_bets > 0 THEN ROUND(total_wagered / total_bets, 2)
        ELSE 0 
    END as avg_bet_size,
    CASE 
        WHEN daily_active_users > 0 THEN ROUND(total_sessions::DECIMAL / daily_active_users, 2)
        ELSE 0 
    END as sessions_per_user
    
FROM daily_activity
ORDER BY activity_date;

-- Customer Segmentation Analysis
-- =============================================
CREATE VIEW v_customer_segmentation AS
WITH customer_behavior AS (
    SELECT 
        c.customer_id,
        c.customer_segment as original_segment,
        c.acquisition_channel,
        c.signup_date,
        
        -- Betting behavior
        COUNT(ba.bet_id) as total_bets,
        COALESCE(SUM(ba.bet_amount), 0) as total_wagered,
        COALESCE(AVG(ba.bet_amount), 0) as avg_bet_size,
        
        -- Session behavior  
        COUNT(DISTINCT s.session_id) as total_sessions,
        COALESCE(AVG(s.duration_minutes), 0) as avg_session_duration,
        COALESCE(AVG(s.num_bets), 0) as avg_bets_per_session,
        
        -- Financial behavior
        COALESCE(SUM(CASE WHEN t.transaction_type = 'deposit' THEN t.amount END), 0) as total_deposits,
        COUNT(CASE WHEN t.transaction_type = 'deposit' THEN 1 END) as num_deposits,
        
        -- Recency
        MAX(GREATEST(ba.bet_date, t.transaction_date)) as last_activity_date,
        DATE_PART('day', NOW() - MAX(GREATEST(ba.bet_date, t.transaction_date))) as days_since_last_activity
        
    FROM customers c
    LEFT JOIN betting_activity ba ON c.customer_id = ba.customer_id
    LEFT JOIN sessions s ON c.customer_id = s.customer_id  
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id, c.customer_segment, c.acquisition_channel, c.signup_date
),
rfm_scores AS (
    SELECT 
        *,
        -- Recency (lower is better)
        CASE 
            WHEN days_since_last_activity <= 7 THEN 5
            WHEN days_since_last_activity <= 30 THEN 4
            WHEN days_since_last_activity <= 90 THEN 3
            WHEN days_since_last_activity <= 180 THEN 2
            ELSE 1
        END as recency_score,
        
        -- Frequency (higher is better)
        CASE 
            WHEN total_bets >= 100 THEN 5
            WHEN total_bets >= 50 THEN 4
            WHEN total_bets >= 20 THEN 3
            WHEN total_bets >= 5 THEN 2
            WHEN total_bets >= 1 THEN 1
            ELSE 0
        END as frequency_score,
        
        -- Monetary (higher is better)
        CASE 
            WHEN total_wagered >= 5000 THEN 5
            WHEN total_wagered >= 1000 THEN 4
            WHEN total_wagered >= 500 THEN 3
            WHEN total_wagered >= 100 THEN 2
            WHEN total_wagered >= 10 THEN 1
            ELSE 0
        END as monetary_score
        
    FROM customer_behavior
)
SELECT 
    customer_id,
    original_segment,
    acquisition_channel,
    signup_date,
    total_bets,
    total_wagered,
    avg_bet_size,
    total_sessions,
    avg_session_duration,
    total_deposits,
    days_since_last_activity,
    recency_score,
    frequency_score,
    monetary_score,
    
    -- RFM combined score
    recency_score + frequency_score + monetary_score as rfm_score,
    
    -- Behavioral segment
    CASE 
        WHEN total_bets = 0 THEN 'Non-Gambler'
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champion'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customer'
        WHEN recency_score >= 4 AND monetary_score >= 3 THEN 'New Customer'
        WHEN frequency_score >= 4 AND monetary_score >= 4 THEN 'Big Spender'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customer'
        WHEN monetary_score <= 2 THEN 'Low Value'
        ELSE 'Regular'
    END as behavioral_segment,
    
    last_activity_date
    
FROM rfm_scores
ORDER BY rfm_score DESC;

-- Channel Attribution Analysis  
-- =============================================
CREATE VIEW v_channel_attribution AS
WITH customer_journey AS (
    SELECT 
        c.customer_id,
        c.acquisition_channel as first_touch_channel,
        c.acquisition_cost,
        c.signup_date,
        
        -- Last touch before first deposit
        t.transaction_date as first_deposit_date,
        t.amount as first_deposit_amount,
        
        -- Customer value metrics
        ltv.customer_ltv,
        ltv.total_wagered,
        ltv.gross_gaming_revenue
        
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id AND t.is_first_deposit = true
    LEFT JOIN v_customer_ltv_analysis ltv ON c.customer_id = ltv.customer_id
)
SELECT 
    first_touch_channel,
    COUNT(*) as total_customers,
    COUNT(CASE WHEN first_deposit_date IS NOT NULL THEN 1 END) as converted_customers,
    
    -- Conversion metrics
    ROUND(COUNT(CASE WHEN first_deposit_date IS NOT NULL THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as conversion_rate,
    
    -- Cost metrics
    SUM(acquisition_cost) as total_acquisition_cost,
    ROUND(AVG(acquisition_cost), 2) as avg_cpi,
    ROUND(SUM(acquisition_cost) / NULLIF(COUNT(CASE WHEN first_deposit_date IS NOT NULL THEN 1 END), 0), 2) as cpa,
    
    -- Revenue metrics
    COALESCE(SUM(first_deposit_amount), 0) as total_first_deposits,
    COALESCE(SUM(gross_gaming_revenue), 0) as total_ggr,
    COALESCE(SUM(customer_ltv), 0) as total_ltv,
    
    -- Performance ratios
    ROUND(COALESCE(SUM(gross_gaming_revenue), 0) / NULLIF(SUM(acquisition_cost), 0), 2) as roas_ggr,
    ROUND(COALESCE(SUM(customer_ltv), 0) / NULLIF(SUM(acquisition_cost), 0), 2) as ltv_cac_ratio,
    
    -- Averages
    ROUND(AVG(CASE WHEN first_deposit_date IS NOT NULL THEN first_deposit_amount END), 2) as avg_first_deposit,
    ROUND(AVG(customer_ltv), 2) as avg_customer_ltv
    
FROM customer_journey
GROUP BY first_touch_channel
ORDER BY total_ggr DESC;

-- Weekly/Monthly Trends
-- =============================================
CREATE VIEW v_weekly_trends AS
WITH weekly_metrics AS (
    SELECT 
        DATE_TRUNC('week', activity_date) as week_start,
        SUM(daily_active_users) as total_weekly_active_users,
        SUM(total_bets) as total_weekly_bets,
        SUM(total_wagered) as total_weekly_wagered,
        SUM(gross_gaming_revenue) as total_weekly_ggr,
        SUM(deposits_amount) as total_weekly_deposits,
        AVG(avg_bet_size) as avg_weekly_bet_size,
        COUNT(*) as days_in_week
    FROM v_daily_engagement
    GROUP BY DATE_TRUNC('week', activity_date)
)
SELECT 
    week_start,
    week_start + INTERVAL '6 days' as week_end,
    total_weekly_active_users,
    ROUND(total_weekly_active_users::DECIMAL / days_in_week, 0) as avg_daily_active_users,
    total_weekly_bets,
    total_weekly_wagered,
    total_weekly_ggr,
    total_weekly_deposits,
    ROUND(avg_weekly_bet_size, 2) as avg_bet_size,
    
    -- Week-over-week growth
    ROUND(
        (total_weekly_ggr - LAG(total_weekly_ggr) OVER (ORDER BY week_start)) / 
        NULLIF(LAG(total_weekly_ggr) OVER (ORDER BY week_start), 0) * 100, 2
    ) as ggr_wow_growth_pct,
    
    ROUND(
        (total_weekly_active_users - LAG(total_weekly_active_users) OVER (ORDER BY week_start)) / 
        NULLIF(LAG(total_weekly_active_users) OVER (ORDER BY week_start), 0) * 100, 2
    ) as users_wow_growth_pct

FROM weekly_metrics
ORDER BY week_start;