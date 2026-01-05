import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


#fixed random seed for reproducibility
RANDOM_SEED = 26
np.random.seed(RANDOM_SEED)

# Load data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

# Train Decision Tree and forecast future inflation
def decision_tree_forecast(df, forecast_start='2026-04-01', forecast_end='2031-04-01', max_depth=5):
    # Use all data up to 2024-12-31 for training
    train_df = df[df['date'] <= '2024-12-31']

    # X = time steps, y = inflation
    X_train = np.arange(len(train_df)).reshape(-1, 1)
    y_train = train_df['inflation_rate'].values

    # Train model
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Determine forecast horizon
    last_index = len(train_df)
    # Create monthly dates for forecast
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='MS')
    X_forecast = np.arange(last_index, last_index + len(forecast_dates)).reshape(-1, 1)

    # Predict
    forecast_values = model.predict(X_forecast)

    # Return as DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_inflation': forecast_values
    })

    return forecast_df

# Example usage
df = load_data("inflation.csv")
forecast_df = decision_tree_forecast(df)

print(forecast_df)

"""
# Scaling Factor for Workforce Simplification
# Actual UK workforce ~ 34 million. Simplified workforce = 100,000.
# All financial inputs must be scaled down by this factor.
ACTUAL_WORKFORCE = 34_200_000
SIMPLIFIED_WORKFORCE = 100_000
SCALING_FACTOR = SIMPLIFIED_WORKFORCE / ACTUAL_WORKFORCE 

# Time Horizon
YEARS = 5 #until the tax freeze is over in 2031

# Spending multiplier time  lag distribution
# Year 1: 20%, Year 2: 30%, Year 3: 25%, Year 4: 15%, Year 5: 10%
LAG_SHARES = np.array([0.20, 0.30, 0.25, 0.15, 0.10])

#The mean values represent the approximate annual financial impact of the stated policies.
#All values are in billions of GBP per year and scaled down for the simplified model.

#policy 1 Increase spending on public services by 32 billion per year
#policy 2: Removal of 2 child benefit cap (Estimated annual cost: 2.5 billion)
# Total Spending: 34.5 billion
SPENDING_PARAMS = {'mean': (32 + 2.5) * SCALING_FACTOR, 'std': 5.0 * SCALING_FACTOR}

#policy 3:Tax per mile driven (Estimated annual revenue of 375 million for EVs and about 125 million for PHEV)
TAX_PER_MILE_PARAMS={'mean': 0.5 , 'std': 0.15}

#policy 4:fiscal drag due to tax freezes (estimated 29.3 billion according to OBR)
FISCAL_DRAG_PARAMS={'mean' : 29.3 * SCALING_FACTOR, 'std': 3.5 * SCALING_FACTOR}

# Keynesian Multiplier and Propensity to Consume
MPC_PARAMS = {'mean': 0.3,'std': 0.2} #taken as 0.3 due  to low consumer confidence in the UK

# Multiplier: K = 1 / (1 - MPC). sample MPC and calculate K from it.
# clip MPC because it is always between 0 and 1

# Multiplier Uncertainty (Used as an alternative if not sampling MPC, but we use MPC)
# MULTIPLIER_PARAMS = {'mean': 1.5, 'std': 0.5}

# Monte Carlo Requirements
N_SIMULATIONS = 20000

#Running the actual simulation

# Array to store results
results = []

for i in range(N_SIMULATIONS):
    #Sample Parameters for the Simulation run
    
    # Sample MPC and calculate Multipliers
    mpc = np.random.normal(MPC_PARAMS['mean'], MPC_PARAMS['std'])
    mpc = np.clip(mpc, 0.01, 0.99) # MPC must be between 0 and 1
    
    kg = 1 / (1 - mpc)          # Spending Multiplier
    kt = -mpc / (1 - mpc)       # Tax Multiplier
    
    # Sample Annual Spending 
    gov_spending_annual = max(0, np.random.normal(SPENDING_PARAMS['mean'], SPENDING_PARAMS['std']))
    
    # Sample Annual Revenue Components
    fiscal_drag_annual = max(0, np.random.normal(FISCAL_DRAG_PARAMS['mean'], FISCAL_DRAG_PARAMS['std'])) * SCALING_FACTOR
    tax_per_mile_annual = max(0, np.random.normal(TAX_PER_MILE_PARAMS['mean'], TAX_PER_MILE_PARAMS['std'])) * SCALING_FACTOR
    

    #year by year aggregation
    total_fiscal_position = 0
    tax_gdp_effect = 0
    spending_gdp_effect = 0
    
    # Loop over the 5 years to apply time-specific policy
    for year in range(1, YEARS + 1): 
        
        #tax Revenue Calculation for year T
        current_year_revenue = fiscal_drag_annual # Fiscal drag is always present
        
        # Tax Per Mile is only active from Year 3 (April 2028 onwards) onwards
        if year >= 3:
            current_year_revenue += tax_per_mile_annual
        
        # Fiscal Position = Revenue - Spending (T - G)
        annual_fiscal_position = current_year_revenue - gov_spending_annual
        total_fiscal_position += annual_fiscal_position
        
        #GDP effect
        #tax GDP effect = revenue * KT
        # Assumed to be immediate (no lag applied to withdrawal)
        tax_gdp_effect += current_year_revenue * kt
        
        #Spending GDP Effect (Expansionary Injection with Time Lag)
        delta_G = gov_spending_annual
        base_gdp_impact = delta_G * kg
        
        # Determine which lag shares apply to this year's spending
        # We look forward from the current year.
        lags_to_apply = LAG_SHARES[:YEARS - year + 1]
        total_lag_share = lags_to_apply.sum()
        
        spending_gdp_effect += base_gdp_impact * total_lag_share
        
    # Net GDP Effect = Expansionary Force (G) + Contractionary Force (T)
    total_net_gdp_effect = spending_gdp_effect + tax_gdp_effect

    # Store the results
    results.append({'Sim_ID': i, 'MPC': mpc, 'K_G': kg, 'K_T': kt, 'Annual_Spending': gov_spending_annual,
    'Total_5Y_Fiscal_Position': total_fiscal_position, 'Total_5Y_GDP_Effect': total_net_gdp_effect })

#store results list to a DataFrame
df_results = pd.DataFrame(results)

#gnenrate outputs
#rescale the results back to actual worfkorce of 34 million
df_results_actual = df_results.copy()
df_results_actual['Total_5Y_Fiscal_Position_Actual'] = df_results_actual['Total_5Y_Fiscal_Position'] / SCALING_FACTOR
df_results_actual['Total_5Y_GDP_Effect_Actual'] = df_results_actual['Total_5Y_GDP_Effect'] / SCALING_FACTOR

#Summary Statistics

total_fiscal = df_results_actual['Total_5Y_Fiscal_Position_Actual']

#Mean
mean_fiscal = total_fiscal.mean()
mean_gdp = df_results_actual['Total_5Y_GDP_Effect_Actual'].mean()

#Median
median_fiscal = total_fiscal.median()

#Probability of a surplus (Fiscal Position > 0)
prob_surplus = (total_fiscal > 0).mean() * 100

print("Cumulative 5 year impact in Billion of pounds")
print("---")
print(f"Total 5-Year Net Fiscal Position (surplus or deficit):")
print(f" Mean: {mean_fiscal:,.2f} B£")
print(f" Median: {median_fiscal:,.2f} B£")
print(f" Probability of a Surplus: {prob_surplus:.2f}%")
print(f"\nNet GDP effevt:")
print(f" Mean GDP Boost: {mean_gdp:,.2f} B£")


# Histogram of the Net Fiscal Position
plt.figure(figsize=(10, 6))
df_results_actual['Total_5Y_Fiscal_Position_Actual'].hist(bins=50, density=True, color='#1f77b4', edgecolor='black', alpha=0.7)
total_fiscal.plot(kind='kde', color='red', linewidth=2)
# Mark the mean and zero line
plt.axvline(mean_fiscal, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_fiscal:.2f} B£')
plt.axvline(0, color='k', linestyle='-', linewidth=1, label='Zero/Break-Even')
#add title, labels to axes, and legend 
plt.title('Monte Carlo Simulation: Distribution of Net 5-Year Fiscal Position', fontsize=14)
plt.xlabel('Net Fiscal Position (Cumulative 5 Years, Billions GBP)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.savefig('Net_Fiscal_Postion.png')
plt.show()
"""