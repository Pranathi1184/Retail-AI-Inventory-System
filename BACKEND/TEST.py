# advanced_retail_predictor_with_input.py
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedRetailPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessing_objects = {}
        self.results = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Load individual models
            model_files = {
                'random_forest': 'random_forest_model.pkl',
                'xgboost': 'xgboost_model.pkl', 
                'lightgbm': 'lightgbm_model.pkl',
                'catboost': 'catboost_model.pkl'
            }
            
            for name, file_path in model_files.items():
                with open(file_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
            
            # Load preprocessing objects
            with open('preprocessing_objects.pkl', 'rb') as f:
                self.preprocessing_objects = pickle.load(f)
            
            # Load results for weights
            with open('model_results.json', 'r') as f:
                self.results = json.load(f)
            
            print("✅ Advanced Retail Predictor Initialized!")
            print(f"📊 Available models: {list(self.models.keys())}")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please train the models first using the training script.")
    
    def get_user_input(self):
        """Get product data from user input"""
        print("\n🛍️  Please enter product details:")
        print("=" * 50)
        
        product_data = {}
        
        # Basic product information
        product_data['Date'] = input("📅 Date (YYYY-MM-DD) [default: today]: ").strip()
        if not product_data['Date']:
            product_data['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        product_data['Store ID'] = input("🏪 Store ID (e.g., S001, S002): ").strip()
        product_data['Product ID'] = input("📦 Product ID (e.g., P0001, P0002): ").strip()
        product_data['Category'] = input("📋 Category (Groceries/Toys/Electronics/Clothing/Furniture): ").strip()
        product_data['Region'] = input("🌍 Region (North/South/East/West): ").strip()
        
        # Inventory and sales data
        print("\n📊 Inventory & Sales Data:")
        product_data['Inventory Level'] = float(input("📦 Current Inventory Level: ") or 100)
        product_data['Units Ordered'] = float(input("📥 Units Ordered (recent): ") or 50)
        product_data['Units Sold'] = float(input("💰 Units Sold (recent): ") or 40)
        
        # Pricing information
        print("\n💰 Pricing Information:")
        product_data['Price'] = float(input("🏷️  Current Price: ") or 30.0)
        product_data['Discount'] = float(input("🎯 Current Discount (%): ") or 10)
        product_data['Competitor Pricing'] = float(input("⚔️  Competitor Price: ") or 28.0)
        
        # External factors
        print("\n🌤️  External Factors:")
        product_data['Weather Condition'] = input("☀️  Weather (Sunny/Rainy/Cloudy/Snowy): ").strip()
        product_data['Holiday/Promotion'] = int(input("🎉 Holiday or Promotion? (0=No, 1=Yes): ") or 0)
        product_data['Seasonality'] = input("🍂 Season (Winter/Spring/Summer/Autumn): ").strip()
        
        # Demand forecast (if available)
        product_data['Demand Forecast'] = float(input("📈 Current Demand Forecast (if known, else press Enter): ") or product_data['Units Sold'])
        
        print("\n✅ Product data collected successfully!")
        return product_data
    
    def validate_user_input(self, product_data):
        """Validate and clean user input"""
        # Ensure numeric fields are properly formatted
        numeric_fields = ['Inventory Level', 'Units Ordered', 'Units Sold', 'Price', 
                         'Discount', 'Competitor Pricing', 'Demand Forecast']
        
        for field in numeric_fields:
            if field in product_data:
                try:
                    product_data[field] = float(product_data[field])
                except (ValueError, TypeError):
                    product_data[field] = 0.0
        
        # Ensure categorical fields have valid values
        if product_data['Category'] not in ['Groceries', 'Toys', 'Electronics', 'Clothing', 'Furniture']:
            print(f"⚠️  Invalid category. Defaulting to 'Groceries'")
            product_data['Category'] = 'Groceries'
        
        if product_data['Region'] not in ['North', 'South', 'East', 'West']:
            print(f"⚠️  Invalid region. Defaulting to 'North'")
            product_data['Region'] = 'North'
        
        if product_data['Weather Condition'] not in ['Sunny', 'Rainy', 'Cloudy', 'Snowy']:
            print(f"⚠️  Invalid weather. Defaulting to 'Sunny'")
            product_data['Weather Condition'] = 'Sunny'
        
        if product_data['Seasonality'] not in ['Winter', 'Spring', 'Summer', 'Autumn']:
            print(f"⚠️  Invalid season. Defaulting to 'Winter'")
            product_data['Seasonality'] = 'Winter'
        
        return product_data
    
    def create_advanced_features(self, df):
        """Create advanced features for better predictions"""
        # Sales velocity (recent sales trend)
        if 'Units Sold' in df.columns:
            df['sales_velocity_7d'] = df['Units Sold'].rolling(7, min_periods=1).mean()
            df['sales_velocity_30d'] = df['Units Sold'].rolling(30, min_periods=1).mean()
        
        # Price elasticity indicators
        if all(col in df.columns for col in ['Price', 'Units Sold']):
            df['price_ratio'] = df['Price'] / df['Competitor Pricing']
            df['discount_effectiveness'] = df['Discount'] * df['Units Sold']
        
        # Seasonal patterns
        if 'month' in df.columns:
            df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Inventory turnover
        if all(col in df.columns for col in ['Inventory Level', 'Units Sold']):
            df['inventory_turnover'] = df['Units Sold'] / (df['Inventory Level'] + 1)
        
        return df
    
    def predict_demand_forecast(self, input_data, forecast_days=30):
        """Predict demand forecast for the next N days"""
        print(f"📈 Predicting demand forecast for {forecast_days} days...")
        
        # Prepare features for demand prediction
        processed_data = self.preprocess_new_data(input_data.copy())
        processed_data = self.create_advanced_features(processed_data)
        processed_data = self.create_lag_features_for_prediction(processed_data)
        
        X_new, feature_columns = self.prepare_features_for_prediction(processed_data)
        
        # Get predictions from all models
        demand_predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    X_new_clean = X_new.copy()
                    X_new_clean.columns = [col.replace(' ', '_').replace('-', '_') for col in X_new_clean.columns]
                    pred = model.predict(X_new_clean)
                else:
                    pred = model.predict(X_new)
                
                demand_predictions[name] = pred
                
            except Exception as e:
                print(f"❌ Error in {name} demand prediction: {e}")
                demand_predictions[name] = np.zeros(len(X_new))
        
        # Create hybrid demand forecast
        if 'hybrid' in self.results:
            weights = self.results['hybrid']['weights']
            hybrid_demand = np.zeros(len(X_new), dtype=np.float64)
            
            for name, weight in weights.items():
                if name in demand_predictions:
                    hybrid_demand += weight * demand_predictions[name].astype(np.float64)
        
        # Generate multi-day forecast
        base_demand = hybrid_demand[0] if 'hybrid' in self.results else np.mean(list(demand_predictions.values()))
        daily_forecasts = self.generate_multi_day_forecast(base_demand, forecast_days, processed_data)
        
        return {
            'immediate_demand': base_demand,
            'daily_forecasts': daily_forecasts,
            'total_30day_forecast': sum(daily_forecasts.values()),
            'confidence_score': self.results['hybrid']['r2'] if 'hybrid' in self.results else 0.5
        }
    
    def generate_multi_day_forecast(self, base_demand, days, processed_data):
        """Generate multi-day demand forecast with trends"""
        forecasts = {}
        current_date = datetime.now()
        seasonal_factors = self.get_seasonal_factors(processed_data)
        
        for i in range(days):
            forecast_date = current_date + timedelta(days=i)
            day_of_week = forecast_date.weekday()
            month = forecast_date.month
            
            weekday_factor = 0.9 if day_of_week >= 5 else 1.1
            seasonal_factor = seasonal_factors.get(month, 1.0)
            
            daily_demand = base_demand * weekday_factor * seasonal_factor
            variation = np.random.normal(1, 0.1)
            daily_demand *= max(0.7, min(1.3, variation))
            
            forecasts[forecast_date.strftime('%Y-%m-%d')] = max(0, round(daily_demand, 2))
        
        return forecasts
    
    def get_seasonal_factors(self, processed_data):
        """Get seasonal adjustment factors"""
        seasonal_patterns = {
            1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
            7: 1.15, 8: 1.1, 9: 1.0, 10: 1.05, 11: 1.2, 12: 1.3
        }
        return seasonal_patterns
    
    def optimize_pricing(self, input_data, demand_forecast):
        """Optimize pricing based on demand forecast"""
        print("💰 Optimizing pricing strategy...")
        
        base_price = input_data['Price']
        competitor_price = input_data.get('Competitor Pricing', base_price)
        current_discount = input_data.get('Discount', 0)
        demand_level = demand_forecast['immediate_demand']
        
        # Define demand thresholds
        low_demand_threshold = 50
        high_demand_threshold = 150
        
        pricing_strategy = {
            'current_price': base_price,
            'current_discount': current_discount,
            'competitor_price': competitor_price,
            'demand_level': demand_level,
        }
        
        # Strategy 1: Demand-based pricing
        if demand_level < low_demand_threshold:
            recommended_discount = min(30, current_discount + 10)
            strategy = "PROMOTIONAL_PRICING"
            reasoning = f"Low demand ({demand_level:.0f} units). Increase discounts to stimulate sales."
        elif demand_level > high_demand_threshold:
            recommended_discount = max(0, current_discount - 5)
            strategy = "PREMIUM_PRICING"
            reasoning = f"High demand ({demand_level:.0f} units). Reduce discounts for better margins."
        else:
            recommended_discount = current_discount
            strategy = "COMPETITIVE_PRICING"
            reasoning = f"Moderate demand ({demand_level:.0f} units). Maintain competitive positioning."
        
        # Strategy 2: Competitor-based adjustment
        price_ratio = base_price / competitor_price
        if price_ratio > 1.1:
            strategy += "_WITH_PRICE_MATCH"
            reasoning += " Consider matching competitor prices."
            recommended_discount = max(recommended_discount, 15)
        
        optimal_price = base_price * (1 - recommended_discount/100)
        
        pricing_strategy.update({
            'recommended_price': round(optimal_price, 2),
            'recommended_discount': recommended_discount,
            'pricing_strategy': strategy,
            'reasoning': reasoning,
            'expected_impact': self.estimate_pricing_impact(demand_level, optimal_price, base_price)
        })
        
        return pricing_strategy
    
    def estimate_pricing_impact(self, demand, new_price, old_price):
        """Estimate impact of price change"""
        price_change_pct = (new_price - old_price) / old_price
        elasticity = -1.5
        volume_change_pct = elasticity * price_change_pct
        
        old_revenue = demand * old_price
        new_demand = demand * (1 + volume_change_pct)
        new_revenue = new_demand * new_price
        
        revenue_change_pct = (new_revenue - old_revenue) / old_revenue * 100
        
        return {
            'expected_volume_change': round(volume_change_pct * 100, 1),
            'expected_revenue_change': round(revenue_change_pct, 1),
            'old_revenue': round(old_revenue, 2),
            'new_revenue': round(new_revenue, 2)
        }
    
    def calculate_reorder_recommendation(self, input_data, demand_forecast):
        """Calculate optimal reorder quantities"""
        print("📦 Calculating reorder recommendations...")
        
        current_inventory = input_data['Inventory Level']
        lead_time_days = 7
        safety_stock_days = 3
        immediate_demand = demand_forecast['immediate_demand']
        daily_forecasts = demand_forecast['daily_forecasts']
        
        days_of_supply = current_inventory / immediate_demand if immediate_demand > 0 else 999
        lead_time_demand = sum(list(daily_forecasts.values())[:lead_time_days])
        reorder_point = lead_time_demand + (safety_stock_days * immediate_demand)
        eoq = self.calculate_eoq(immediate_demand, input_data)
        
        if current_inventory <= reorder_point:
            urgency = "HIGH"
            action = "IMMEDIATE_REORDER"
            reasoning = f"Current inventory ({current_inventory}) below reorder point ({reorder_point:.0f})"
        elif days_of_supply <= lead_time_days + safety_stock_days:
            urgency = "MEDIUM"
            action = "SCHEDULE_REORDER"
            reasoning = f"Low days of supply ({days_of_supply:.1f} days)"
        else:
            urgency = "LOW"
            action = "MONITOR"
            reasoning = f"Sufficient inventory ({days_of_supply:.1f} days supply)"
        
        return {
            'current_inventory': current_inventory,
            'days_of_supply': round(days_of_supply, 1),
            'reorder_point': round(reorder_point, 0),
            'lead_time_demand': round(lead_time_demand, 0),
            'recommended_order_quantity': round(eoq, 0),
            'reorder_urgency': urgency,
            'reorder_action': action,
            'reasoning': reasoning,
            'stockout_risk': self.calculate_stockout_risk(current_inventory, daily_forecasts)
        }
    
    def calculate_eoq(self, demand, input_data):
        """Calculate Economic Order Quantity"""
        ordering_cost = 50
        holding_cost_rate = 0.25
        unit_cost = input_data.get('Price', 30)
        
        annual_demand = demand * 365
        holding_cost_per_unit = unit_cost * holding_cost_rate
        
        if holding_cost_per_unit > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        else:
            eoq = demand * 30
        
        return min(eoq, demand * 90)
    
    def calculate_stockout_risk(self, current_inventory, daily_forecasts):
        """Calculate probability of stockout"""
        cumulative_demand = 0
        stockout_day = None
        
        for day, demand in daily_forecasts.items():
            cumulative_demand += demand
            if cumulative_demand > current_inventory and stockout_day is None:
                stockout_day = day
        
        if stockout_day:
            return f"High risk - Stockout expected around {stockout_day}"
        else:
            return "Low risk - Sufficient inventory for 30+ days"
    
    def generate_business_alerts(self, input_data, demand_forecast, pricing_strategy, reorder_recommendation):
        """Generate business alerts"""
        alerts = []
        
        if reorder_recommendation['reorder_urgency'] == "HIGH":
            alerts.append({
                'type': 'STOCKOUT_RISK',
                'severity': 'HIGH',
                'message': f"🚨 IMMEDIATE REORDER NEEDED for {input_data['Product ID']}",
                'details': reorder_recommendation['reasoning'],
                'action': f"Order {reorder_recommendation['recommended_order_quantity']} units immediately"
            })
        
        pricing_impact = pricing_strategy['expected_impact']
        if pricing_impact['expected_revenue_change'] > 5:
            alerts.append({
                'type': 'PRICING_OPPORTUNITY',
                'severity': 'MEDIUM',
                'message': f"💰 Pricing optimization opportunity",
                'details': f"Expected revenue increase: {pricing_impact['expected_revenue_change']}%",
                'action': f"Adjust price to ${pricing_strategy['recommended_price']}"
            })
        
        if demand_forecast['immediate_demand'] > 200:
            alerts.append({
                'type': 'HIGH_DEMAND',
                'severity': 'MEDIUM',
                'message': f"📈 High demand detected for {input_data['Product ID']}",
                'details': f"Current demand: {demand_forecast['immediate_demand']:.0f} units",
                'action': "Consider increasing inventory and reviewing pricing"
            })
        
        if demand_forecast['immediate_demand'] < 20:
            alerts.append({
                'type': 'LOW_PERFORMANCE',
                'severity': 'LOW',
                'message': f"📉 Low demand for {input_data['Product ID']}",
                'details': f"Current demand: {demand_forecast['immediate_demand']:.0f} units",
                'action': "Consider promotions or discontinuation"
            })
        
        return alerts
    
    def comprehensive_analysis(self, input_data):
        """Run comprehensive analysis"""
        print("🔍 Running comprehensive retail analysis...")
        print("=" * 60)
        
        # Validate input data
        input_data = self.validate_user_input(input_data)
        
        # Run all predictions
        demand_forecast = self.predict_demand_forecast(input_data)
        pricing_strategy = self.optimize_pricing(input_data, demand_forecast)
        reorder_recommendation = self.calculate_reorder_recommendation(input_data, demand_forecast)
        business_alerts = self.generate_business_alerts(input_data, demand_forecast, pricing_strategy, reorder_recommendation)
        
        # Compile report
        comprehensive_report = {
            'product_info': {
                'store_id': input_data.get('Store ID', 'Unknown'),
                'product_id': input_data.get('Product ID', 'Unknown'),
                'category': input_data.get('Category', 'Unknown'),
                'region': input_data.get('Region', 'Unknown'),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'demand_forecast': demand_forecast,
            'pricing_strategy': pricing_strategy,
            'inventory_management': reorder_recommendation,
            'business_alerts': business_alerts,
            'overall_health_score': self.calculate_health_score(demand_forecast, reorder_recommendation)
        }
        
        return comprehensive_report
    
    def calculate_health_score(self, demand_forecast, reorder_recommendation):
        """Calculate overall product health score"""
        score = 50
        demand = demand_forecast['immediate_demand']
        
        if demand > 100:
            score += 30
        elif demand > 50:
            score += 20
        elif demand > 20:
            score += 10
        
        urgency = reorder_recommendation['reorder_urgency']
        if urgency == "LOW":
            score += 20
        elif urgency == "MEDIUM":
            score += 10
        
        confidence = demand_forecast['confidence_score']
        score += confidence * 30
        
        return min(100, max(0, round(score)))
    
    # Helper methods
    def preprocess_new_data(self, new_data):
        """Preprocess new data for prediction"""
        processed_data = new_data.copy()
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        processed_data['day_of_week'] = processed_data['Date'].dt.dayofweek
        processed_data['day_of_month'] = processed_data['Date'].dt.day
        processed_data['month'] = processed_data['Date'].dt.month
        processed_data['quarter'] = processed_data['Date'].dt.quarter
        processed_data['year'] = processed_data['Date'].dt.year
        processed_data['is_weekend'] = (processed_data['day_of_week'] >= 5).astype(int)
        
        categorical_columns = ['Store ID', 'Product ID', 'Category', 'Region', 
                             'Weather Condition', 'Seasonality']
        
        for col in categorical_columns:
            if col in processed_data.columns and col in self.preprocessing_objects['label_encoders']:
                le = self.preprocessing_objects['label_encoders'][col]
                processed_data[col] = processed_data[col].astype(str)
                unseen_mask = ~processed_data[col].isin(le.classes_)
                if unseen_mask.any():
                    most_frequent = le.classes_[0]
                    processed_data.loc[unseen_mask, col] = most_frequent
                processed_data[col] = le.transform(processed_data[col])
        
        return processed_data
    
    def create_lag_features_for_prediction(self, df):
        """Create lag features for prediction"""
        lag_days = [1, 3, 7, 14, 30]
        for lag in lag_days:
            df[f'units_sold_lag_{lag}'] = df['Units Sold'].mean() if 'Units Sold' in df.columns else 0
            df[f'inventory_lag_{lag}'] = df['Inventory Level'].mean() if 'Inventory Level' in df.columns else 0
        
        for window in [7, 14, 30]:
            df[f'units_sold_rolling_mean_{window}'] = df['Units Sold'].mean() if 'Units Sold' in df.columns else 0
            df[f'units_sold_rolling_std_{window}'] = df['Units Sold'].std() if 'Units Sold' in df.columns else 1
        
        return df
    
    def prepare_features_for_prediction(self, processed_data):
        """Prepare features in the exact same order as training"""
        feature_columns = [
            'Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level',
            'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 
            'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality',
            'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_weekend',
            'units_sold_lag_1', 'inventory_lag_1', 'units_sold_lag_3', 'inventory_lag_3', 
            'units_sold_lag_7', 'inventory_lag_7', 'units_sold_lag_14', 'inventory_lag_14', 
            'units_sold_lag_30', 'inventory_lag_30', 'units_sold_rolling_mean_7', 
            'units_sold_rolling_std_7', 'units_sold_rolling_mean_14', 'units_sold_rolling_std_14', 
            'units_sold_rolling_mean_30', 'units_sold_rolling_std_30'
        ]
        
        available_features = [col for col in feature_columns if col in processed_data.columns]
        X_new = processed_data[available_features]
        
        for col in feature_columns:
            if col not in available_features:
                X_new[col] = 0
        
        X_new = X_new[feature_columns]
        
        return X_new, feature_columns

def display_comprehensive_report(report):
    """Display the comprehensive analysis report"""
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE RETAIL ANALYSIS REPORT")
    print("=" * 80)
    
    # Product Information
    product_info = report['product_info']
    print(f"\n📋 PRODUCT INFORMATION:")
    print(f"   Store: {product_info['store_id']} | Product: {product_info['product_id']}")
    print(f"   Category: {product_info['category']} | Region: {product_info['region']}")
    print(f"   Analysis Time: {product_info['analysis_timestamp']}")
    print(f"   Overall Health Score: {report['overall_health_score']}/100")
    
    # Demand Forecast
    demand = report['demand_forecast']
    print(f"\n📈 DEMAND FORECAST:")
    print(f"   Immediate Demand: {demand['immediate_demand']:.0f} units")
    print(f"   30-Day Forecast: {demand['total_30day_forecast']:.0f} units")
    print(f"   Confidence: {demand['confidence_score']:.1%}")
    
    print(f"\n   Daily Forecast (next 7 days):")
    for i, (date, forecast) in enumerate(list(demand['daily_forecasts'].items())[:7]):
        print(f"     {date}: {forecast:.0f} units")
    
    # Pricing Strategy
    pricing = report['pricing_strategy']
    print(f"\n💰 PRICING STRATEGY:")
    print(f"   Current Price: ${pricing['current_price']} ({pricing['current_discount']}% discount)")
    print(f"   Recommended Price: ${pricing['recommended_price']} ({pricing['recommended_discount']}% discount)")
    print(f"   Strategy: {pricing['pricing_strategy']}")
    print(f"   Reasoning: {pricing['reasoning']}")
    
    impact = pricing['expected_impact']
    print(f"   Expected Impact:")
    print(f"     Volume Change: {impact['expected_volume_change']}%")
    print(f"     Revenue Change: {impact['expected_revenue_change']}%")
    print(f"     Revenue: ${impact['old_revenue']} → ${impact['new_revenue']}")
    
    # Inventory Management
    inventory = report['inventory_management']
    print(f"\n📦 INVENTORY MANAGEMENT:")
    print(f"   Current Inventory: {inventory['current_inventory']} units")
    print(f"   Days of Supply: {inventory['days_of_supply']} days")
    print(f"   Reorder Point: {inventory['reorder_point']:.0f} units")
    print(f"   Urgency: {inventory['reorder_urgency']} - {inventory['reorder_action']}")
    print(f"   Recommended Order: {inventory['recommended_order_quantity']:.0f} units")
    print(f"   Stockout Risk: {inventory['stockout_risk']}")
    
    # Business Alerts
    alerts = report['business_alerts']
    if alerts:
        print(f"\n🚨 BUSINESS ALERTS:")
        for alert in alerts:
            print(f"   [{alert['severity']}] {alert['type']}: {alert['message']}")
            print(f"      Details: {alert['details']}")
            print(f"      Action: {alert['action']}")
    else:
        print(f"\n✅ No critical alerts at this time.")
    
    print("\n" + "=" * 80)
    print("📊 RECOMMENDED ACTIONS SUMMARY:")
    print("=" * 80)
    
    # Summary of recommended actions
    actions = []
    
    if pricing['recommended_discount'] != pricing['current_discount']:
        actions.append(f"• Adjust pricing to ${pricing['recommended_price']} ({pricing['recommended_discount']}% discount)")
    
    if inventory['reorder_urgency'] in ['HIGH', 'MEDIUM']:
        actions.append(f"• Order {inventory['recommended_order_quantity']:.0f} units of {product_info['product_id']}")
    
    if demand['immediate_demand'] > 150:
        actions.append("• Monitor stock levels closely - high demand period")
    elif demand['immediate_demand'] < 30:
        actions.append("• Consider promotional activities to boost sales")
    
    if not actions:
        actions.append("• Maintain current strategy - performance is optimal")
    
    for action in actions:
        print(f"   {action}")
    
    return report

def main():
    """Main function to run the interactive predictor"""
    print("🚀 Advanced Retail Prediction System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AdvancedRetailPredictor()
    
    while True:
        print("\n" + "=" * 50)
        print("🛍️  MAIN MENU")
        print("=" * 50)
        print("1. Analyze Single Product (User Input)")
        print("2. Analyze Sample Product")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            # Get user input
            product_data = predictor.get_user_input()
            
            # Run analysis
            report = predictor.comprehensive_analysis(product_data)
            
            # Display results
            display_comprehensive_report(report)
            
            # Save report
            filename = f"analysis_{product_data['Product ID']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"💾 Report saved as: {filename}")
            
        elif choice == '2':
            # Sample analysis
            sample_data = {
                'Date': '2024-01-15',
                'Store ID': 'S002',
                'Product ID': 'P0001',
                'Category': 'Groceries',
                'Region': 'South',
                'Inventory Level': 200,
                'Units Ordered': 50,
                'Demand Forecast': 40,
                'Price': 30.0,
                'Discount': 10,
                'Weather Condition': 'Rainy',
                'Holiday/Promotion': 0,
                'Competitor Pricing': 28.0,
                'Seasonality': 'Winter',
                'Units Sold': 35
            }
            
            print("\n📊 Analyzing Sample Product...")
            report = predictor.comprehensive_analysis(sample_data)
            display_comprehensive_report(report)
            
        elif choice == '3':
            print("👋 Thank you for using Advanced Retail Predictor!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to analyze another product? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Thank you for using Advanced Retail Predictor!")
            break

if __name__ == "__main__":
    main()