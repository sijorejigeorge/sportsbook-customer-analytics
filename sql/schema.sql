-- =============================================
-- Database Setup and Table Creation
-- =============================================
-- This script creates the database schema for the sportsbook
-- marketing analytics system using PostgreSQL/DuckDB syntax

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS customer_campaigns CASCADE;
DROP TABLE IF EXISTS campaigns CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS betting_activity CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Customers table
-- =============================================
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    signup_date DATE NOT NULL,
    acquisition_channel VARCHAR(50) NOT NULL,
    acquisition_cost DECIMAL(10,2) DEFAULT 0,
    age INTEGER,
    gender CHAR(1),
    country VARCHAR(3),
    state VARCHAR(10),
    city VARCHAR(100),
    customer_segment VARCHAR(20),
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table
-- =============================================
CREATE TABLE transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('deposit', 'withdrawal')),
    amount DECIMAL(10,2) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    payment_method VARCHAR(30),
    is_first_deposit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Betting Activity table
-- =============================================
CREATE TABLE betting_activity (
    bet_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    session_id VARCHAR(30),
    sport VARCHAR(20),
    bet_amount DECIMAL(10,2) NOT NULL,
    bet_date TIMESTAMP NOT NULL,
    bet_won BOOLEAN,
    payout DECIMAL(10,2) DEFAULT 0,
    net_result DECIMAL(10,2),
    odds DECIMAL(8,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table  
-- =============================================
CREATE TABLE sessions (
    session_id VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_minutes INTEGER,
    num_bets INTEGER DEFAULT 0,
    total_wagered DECIMAL(10,2) DEFAULT 0,
    net_result DECIMAL(10,2) DEFAULT 0,
    device_type VARCHAR(20),
    browser VARCHAR(30),
    country VARCHAR(3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Campaigns table
-- =============================================
CREATE TABLE campaigns (
    campaign_id VARCHAR(20) PRIMARY KEY,
    campaign_name VARCHAR(200) NOT NULL,
    campaign_type VARCHAR(30),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    description TEXT,
    cost_per_customer DECIMAL(8,2),
    bonus_amount DECIMAL(8,2),
    min_deposit DECIMAL(8,2),
    target_segments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer Campaigns table (many-to-many relationship)
-- =============================================
CREATE TABLE customer_campaigns (
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    campaign_id VARCHAR(20) NOT NULL REFERENCES campaigns(campaign_id),
    sent_date DATE NOT NULL,
    responded BOOLEAN DEFAULT FALSE,
    response_date DATE,
    bonus_claimed BOOLEAN DEFAULT FALSE,
    campaign_cost DECIMAL(8,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, campaign_id)
);

-- Create indexes for better query performance
-- =============================================

-- Customers indexes
CREATE INDEX idx_customers_signup_date ON customers(signup_date);
CREATE INDEX idx_customers_channel ON customers(acquisition_channel);
CREATE INDEX idx_customers_segment ON customers(customer_segment);

-- Transactions indexes
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_first_deposit ON transactions(is_first_deposit);

-- Betting activity indexes
CREATE INDEX idx_betting_customer_id ON betting_activity(customer_id);
CREATE INDEX idx_betting_date ON betting_activity(bet_date);
CREATE INDEX idx_betting_sport ON betting_activity(sport);
CREATE INDEX idx_betting_session ON betting_activity(session_id);

-- Sessions indexes
CREATE INDEX idx_sessions_customer_id ON sessions(customer_id);
CREATE INDEX idx_sessions_start_time ON sessions(start_time);
CREATE INDEX idx_sessions_device ON sessions(device_type);

-- Campaigns indexes
CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);
CREATE INDEX idx_campaigns_type ON campaigns(campaign_type);

-- Customer campaigns indexes
CREATE INDEX idx_customer_campaigns_sent_date ON customer_campaigns(sent_date);
CREATE INDEX idx_customer_campaigns_responded ON customer_campaigns(responded);

-- Add foreign key constraints for referential integrity
-- =============================================

-- Add session_id foreign key to betting_activity (optional, sessions created after bets)
-- ALTER TABLE betting_activity ADD CONSTRAINT fk_betting_session 
--     FOREIGN KEY (session_id) REFERENCES sessions(session_id);

-- Create sample data loading script
-- =============================================

-- Function to load CSV data (PostgreSQL specific)
CREATE OR REPLACE FUNCTION load_sportsbook_data(data_path TEXT DEFAULT 'data/raw/')
RETURNS TEXT AS $$
BEGIN
    -- Load customers
    EXECUTE format('COPY customers FROM ''%scustomers.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    -- Load transactions  
    EXECUTE format('COPY transactions FROM ''%stransactions.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    -- Load betting activity
    EXECUTE format('COPY betting_activity FROM ''%sbetting_activity.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    -- Load sessions
    EXECUTE format('COPY sessions FROM ''%ssessions.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    -- Load campaigns
    EXECUTE format('COPY campaigns FROM ''%scampaigns.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    -- Load customer campaigns
    EXECUTE format('COPY customer_campaigns FROM ''%scustomer_campaigns.csv'' DELIMITER '','' CSV HEADER', data_path);
    
    RETURN 'Data loaded successfully';
END;
$$ LANGUAGE plpgsql;

-- Basic data validation queries
-- =============================================

-- Data quality checks
CREATE VIEW v_data_quality_summary AS
SELECT 
    'customers' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_ids,
    COUNT(CASE WHEN acquisition_cost < 0 THEN 1 END) as negative_costs,
    MIN(signup_date) as earliest_date,
    MAX(signup_date) as latest_date
FROM customers

UNION ALL

SELECT 
    'transactions' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_customer_ids,
    COUNT(CASE WHEN amount <= 0 THEN 1 END) as invalid_amounts,
    MIN(transaction_date::date) as earliest_date,
    MAX(transaction_date::date) as latest_date
FROM transactions

UNION ALL

SELECT 
    'betting_activity' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_customer_ids,
    COUNT(CASE WHEN bet_amount <= 0 THEN 1 END) as invalid_amounts,
    MIN(bet_date::date) as earliest_date,
    MAX(bet_date::date) as latest_date
FROM betting_activity

UNION ALL

SELECT 
    'sessions' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_customer_ids,
    COUNT(CASE WHEN duration_minutes < 0 THEN 1 END) as negative_durations,
    MIN(start_time::date) as earliest_date,
    MAX(start_time::date) as latest_date
FROM sessions;

-- Summary statistics
-- =============================================
CREATE VIEW v_database_summary AS
SELECT 
    (SELECT COUNT(*) FROM customers) as total_customers,
    (SELECT COUNT(*) FROM transactions) as total_transactions,
    (SELECT COUNT(*) FROM betting_activity) as total_bets,
    (SELECT COUNT(*) FROM sessions) as total_sessions,
    (SELECT COUNT(*) FROM campaigns) as total_campaigns,
    (SELECT COUNT(*) FROM customer_campaigns) as total_campaign_interactions,
    (SELECT COUNT(DISTINCT customer_id) FROM transactions WHERE is_first_deposit = true) as converted_customers,
    (SELECT ROUND(SUM(amount), 2) FROM transactions WHERE transaction_type = 'deposit') as total_deposits,
    (SELECT ROUND(SUM(bet_amount), 2) FROM betting_activity) as total_wagered,
    (SELECT ROUND(SUM(bet_amount - payout), 2) FROM betting_activity) as total_ggr;

-- Grant permissions (adjust as needed for your setup)
-- =============================================

-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_user;
-- GRANT SELECT ON ALL VIEWS IN SCHEMA public TO analytics_user;

COMMENT ON TABLE customers IS 'Customer demographic and acquisition information';
COMMENT ON TABLE transactions IS 'Customer deposit and withdrawal transactions';
COMMENT ON TABLE betting_activity IS 'Individual bet records with outcomes';
COMMENT ON TABLE sessions IS 'User session data aggregated from betting activity';
COMMENT ON TABLE campaigns IS 'Marketing campaign definitions';
COMMENT ON TABLE customer_campaigns IS 'Customer campaign participation and responses';