from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)
    alerts = db.relationship('InventoryAlert', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    store_id = db.Column(db.String(50), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    prediction_result = db.Column(db.Text, nullable=False)
    demand_forecast = db.Column(db.Float, nullable=False)
    recommended_price = db.Column(db.Float, nullable=False)
    reorder_quantity = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    alerts = db.relationship('InventoryAlert', backref='prediction', lazy=True)

class InventoryAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=False)
    store_id = db.Column(db.String(50), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # REORDER, PRICING, STOCKOUT, etc.
    severity = db.Column(db.String(20), nullable=False)  # HIGH, MEDIUM, LOW
    message = db.Column(db.String(200), nullable=False)
    details = db.Column(db.Text, nullable=False)
    resolved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

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
                'random_forest': 'models/random_forest_model.pkl',
                'xgboost': 'models/xgboost_model.pkl', 
                'lightgbm': 'models/lightgbm_model.pkl',
                'catboost': 'models/catboost_model.pkl'
            }
            
            for name, file_path in model_files.items():
                try:
                    with open(file_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    print(f"✅ Loaded {name} model")
                except FileNotFoundError:
                    print(f"⚠️  {name} model file not found, using demo mode")
                    self.models[name] = None
            
            # Load preprocessing objects
            try:
                with open('models/preprocessing_objects.pkl', 'rb') as f:
                    self.preprocessing_objects = pickle.load(f)
            except FileNotFoundError:
                print("⚠️  Preprocessing objects not found, using demo mode")
                self.preprocessing_objects = {
                    'label_encoders': {},
                    'scalers': {}
                }
            
            # Load results for weights
            try:
                with open('models/model_results.json', 'r') as f:
                    self.results = json.load(f)
                print(f"✅ Loaded Hybrid model")
            except FileNotFoundError:
                print("⚠️  Model results not found, using default weights")
                self.results = {
                    'hybrid': {
                        'r2': 0.85,
                        'weights': {
                            'random_forest': 0.25,
                            'xgboost': 0.25,
                            'lightgbm': 0.25,
                            'catboost': 0.25
                        }
                    }
                }
            
            print("✅ Advanced Retail Predictor Initialized!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.initialize_demo_mode()
    
    def initialize_demo_mode(self):
        """Initialize demo mode for testing"""
        print("🚀 Initializing Demo Mode...")
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'lightgbm': None,
            'catboost': None
        }
        
        self.preprocessing_objects = {
            'label_encoders': {},
            'scalers': {}
        }
        
        self.results = {
            'hybrid': {
                'r2': 0.85,
                'weights': {
                    'random_forest': 0.25,
                    'xgboost': 0.25,
                    'lightgbm': 0.25,
                    'catboost': 0.25
                }
            }
        }
    
    def preprocess_input_data(self, input_data):
        """Preprocess input data for prediction"""
        # Convert to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Handle date conversion
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Create basic features
        if 'Date' in df.columns:
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced features for prediction"""
        # Sales velocity (simplified for demo)
        if 'Units Sold' in df.columns:
            df['sales_velocity'] = df['Units Sold']
        
        # Price ratio
        if all(col in df.columns for col in ['Price', 'Competitor Pricing']):
            df['price_ratio'] = df['Price'] / (df['Competitor Pricing'] + 0.001)
        
        # Inventory turnover
        if all(col in df.columns for col in ['Inventory Level', 'Units Sold']):
            df['inventory_turnover'] = df['Units Sold'] / (df['Inventory Level'] + 1)
        
        return df
    
    def predict_demand(self, input_data):
        """Predict demand using available models"""
        try:
            # Use demo prediction if no models are available
            if not any(model is not None for model in self.models.values()):
                return self.demo_demand_prediction(input_data)
            
            # Preprocess data
            processed_data = self.preprocess_input_data(input_data)
            processed_data = self.create_advanced_features(processed_data)
            
            # Here you would normally prepare features for your actual models
            # For demo, we'll use the simplified approach
            base_demand = input_data.get('Units Sold', 50) * np.random.uniform(0.8, 1.2)
            
            return max(10, base_demand)  # Ensure minimum demand
            
        except Exception as e:
            print(f"Error in demand prediction: {e}")
            # Fallback to demo prediction
            return self.demo_demand_prediction(input_data)
    
    def demo_demand_prediction(self, input_data):
        """Demo demand prediction when models are not available"""
        base_sales = input_data.get('Units Sold', 50)
        price = input_data.get('Price', 30)
        discount = input_data.get('Discount', 0)
        inventory = input_data.get('Inventory Level', 100)
        holiday = input_data.get('Holiday/Promotion', 0)
        
        # Simple heuristic-based prediction
        demand = base_sales
        
        # Adjust for price (price elasticity)
        price_effect = 1.0 - (price / 100) * 0.1
        demand *= price_effect
        
        # Adjust for discount
        discount_effect = 1.0 + (discount / 100) * 0.5
        demand *= discount_effect
        
        # Adjust for holiday
        if holiday:
            demand *= 1.3
        
        # Add some randomness
        demand *= np.random.uniform(0.9, 1.1)
        
        return max(10, demand)
    
    def optimize_pricing(self, input_data, demand_forecast):
        """Optimize pricing based on demand forecast"""
        current_price = input_data.get('Price', 30)
        competitor_price = input_data.get('Competitor Pricing', current_price * 0.9)
        current_discount = input_data.get('Discount', 0)
        
        # Simple pricing strategy
        price_ratio = current_price / competitor_price
        demand_level = demand_forecast
        
        if demand_level > 100:
            # High demand - can increase price
            recommended_discount = max(0, current_discount - 5)
            strategy = "PREMIUM_PRICING"
            reasoning = "High demand allows for premium pricing"
        elif demand_level < 30:
            # Low demand - need promotions
            recommended_discount = min(30, current_discount + 10)
            strategy = "PROMOTIONAL_PRICING"
            reasoning = "Low demand requires promotional pricing"
        else:
            # Moderate demand - competitive pricing
            if price_ratio > 1.1:
                recommended_discount = min(20, current_discount + 5)
                strategy = "COMPETITIVE_PRICING"
                reasoning = "Price is higher than competition, consider matching"
            else:
                recommended_discount = current_discount
                strategy = "MAINTAIN_PRICING"
                reasoning = "Current pricing is competitive"
        
        optimal_price = current_price * (1 - recommended_discount / 100)
        
        # Calculate expected impact
        price_change_pct = (optimal_price - current_price) / current_price
        elasticity = -1.5  # Assumed price elasticity
        volume_change_pct = elasticity * price_change_pct
        
        old_revenue = demand_level * current_price
        new_demand = demand_level * (1 + volume_change_pct)
        new_revenue = new_demand * optimal_price
        revenue_change_pct = (new_revenue - old_revenue) / old_revenue * 100
        
        return {
            'current_price': current_price,
            'current_discount': current_discount,
            'competitor_price': competitor_price,
            'recommended_price': round(optimal_price, 2),
            'recommended_discount': recommended_discount,
            'pricing_strategy': strategy,
            'reasoning': reasoning,
            'expected_impact': {
                'expected_volume_change': round(volume_change_pct * 100, 1),
                'expected_revenue_change': round(revenue_change_pct, 1),
                'old_revenue': round(old_revenue, 2),
                'new_revenue': round(new_revenue, 2)
            }
        }
    
    def calculate_reorder_recommendation(self, input_data, demand_forecast):
        """Calculate reorder recommendations"""
        current_inventory = input_data.get('Inventory Level', 100)
        immediate_demand = demand_forecast
        
        # Calculate days of supply
        days_of_supply = current_inventory / immediate_demand if immediate_demand > 0 else 999
        
        # Calculate reorder point (7 days of demand + safety stock)
        lead_time_days = 7
        safety_stock_days = 3
        reorder_point = immediate_demand * (lead_time_days + safety_stock_days)
        
        # Calculate EOQ (simplified)
        annual_demand = immediate_demand * 365
        ordering_cost = 50
        holding_cost_rate = 0.25
        unit_cost = input_data.get('Price', 30)
        holding_cost_per_unit = unit_cost * holding_cost_rate
        
        if holding_cost_per_unit > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        else:
            eoq = immediate_demand * 30
        
        # Determine urgency
        if current_inventory <= immediate_demand * lead_time_days:
            urgency = "HIGH"
            action = "IMMEDIATE_REORDER"
            reasoning = f"Current inventory ({current_inventory}) below lead time demand ({immediate_demand * lead_time_days:.0f})"
        elif days_of_supply <= lead_time_days + safety_stock_days:
            urgency = "MEDIUM"
            action = "SCHEDULE_REORDER"
            reasoning = f"Low days of supply ({days_of_supply:.1f} days)"
        else:
            urgency = "LOW"
            action = "MONITOR"
            reasoning = f"Sufficient inventory ({days_of_supply:.1f} days supply)"
        
        # Calculate recommended order quantity
        recommended_quantity = max(0, eoq - current_inventory) if urgency in ['HIGH', 'MEDIUM'] else 0
        
        # Stockout risk assessment
        if days_of_supply < lead_time_days:
            stockout_risk = f"High risk - Stockout expected in {days_of_supply:.0f} days"
        elif days_of_supply < lead_time_days + safety_stock_days:
            stockout_risk = "Medium risk - Monitor closely"
        else:
            stockout_risk = "Low risk - Sufficient inventory"
        
        return {
            'current_inventory': current_inventory,
            'days_of_supply': round(days_of_supply, 1),
            'reorder_point': round(reorder_point, 0),
            'lead_time_demand': round(immediate_demand * lead_time_days, 0),
            'recommended_order_quantity': round(recommended_quantity, 0),
            'reorder_urgency': urgency,
            'reorder_action': action,
            'reasoning': reasoning,
            'stockout_risk': stockout_risk
        }
    
    def generate_dummy_forecast(self, base_demand, days=30):
        """Generate realistic dummy forecast data"""
        forecasts = {}
        current_date = datetime.now()
        
        # Add some seasonality and trends
        for i in range(days):
            date = (current_date + timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Weekly pattern (higher on weekends)
            day_of_week = (current_date + timedelta(days=i)).weekday()
            weekend_factor = 1.2 if day_of_week >= 5 else 1.0
            
            # Random variation
            random_variation = np.random.normal(1, 0.15)
            
            # Calculate daily demand
            daily_demand = base_demand * weekend_factor * random_variation
            forecasts[date] = max(5, round(daily_demand, 2))  # Ensure minimum demand
        
        return forecasts
    
    def generate_business_alerts(self, input_data, demand_forecast, pricing_strategy, reorder_recommendation):
        """Generate business alerts based on analysis"""
        alerts = []
        
        # Inventory alerts
        if reorder_recommendation['reorder_urgency'] == "HIGH":
            alerts.append({
                'type': 'STOCKOUT_RISK',
                'severity': 'HIGH',
                'message': f"🚨 IMMEDIATE REORDER NEEDED for {input_data.get('Product ID', 'Unknown')}",
                'details': reorder_recommendation['reasoning'],
                'action': f"Order {reorder_recommendation['recommended_order_quantity']:.0f} units immediately"
            })
        
        # Pricing alerts
        pricing_impact = pricing_strategy['expected_impact']
        if pricing_impact['expected_revenue_change'] > 5:
            alerts.append({
                'type': 'PRICING_OPPORTUNITY',
                'severity': 'MEDIUM',
                'message': f"💰 Pricing optimization opportunity detected",
                'details': f"Expected revenue increase: {pricing_impact['expected_revenue_change']}%",
                'action': f"Adjust price to ${pricing_strategy['recommended_price']}"
            })
        
        # Demand alerts
        if demand_forecast > 150:
            alerts.append({
                'type': 'HIGH_DEMAND',
                'severity': 'MEDIUM',
                'message': f"📈 High demand detected",
                'details': f"Current demand: {demand_forecast:.0f} units",
                'action': "Consider increasing inventory and reviewing pricing"
            })
        elif demand_forecast < 20:
            alerts.append({
                'type': 'LOW_PERFORMANCE',
                'severity': 'LOW',
                'message': f"📉 Low demand detected",
                'details': f"Current demand: {demand_forecast:.0f} units",
                'action': "Consider promotions or discontinuation review"
            })
        
        return alerts
    
    def comprehensive_analysis(self, input_data):
        """Run comprehensive analysis - main entry point"""
        try:
            print("🔍 Running comprehensive retail analysis...")
            
            # Validate and prepare input data
            validated_data = self.validate_input_data(input_data)
            
            # Predict demand
            immediate_demand = self.predict_demand(validated_data)
            
            # Generate 30-day forecast
            daily_forecasts = self.generate_dummy_forecast(immediate_demand)
            total_30day_forecast = sum(daily_forecasts.values())
            
            # Optimize pricing
            pricing_strategy = self.optimize_pricing(validated_data, immediate_demand)
            
            # Calculate reorder recommendations
            reorder_recommendation = self.calculate_reorder_recommendation(validated_data, immediate_demand)
            
            # Generate business alerts
            business_alerts = self.generate_business_alerts(
                validated_data, immediate_demand, pricing_strategy, reorder_recommendation
            )
            
            # Calculate health score
            health_score = self.calculate_health_score(immediate_demand, reorder_recommendation)
            
            # Compile final report
            report = {
                'product_info': {
                    'store_id': validated_data.get('Store ID', 'Unknown'),
                    'product_id': validated_data.get('Product ID', 'Unknown'),
                    'category': validated_data.get('Category', 'Unknown'),
                    'region': validated_data.get('Region', 'Unknown'),
                    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'demand_forecast': {
                    'immediate_demand': round(immediate_demand, 2),
                    'total_30day_forecast': round(total_30day_forecast, 2),
                    'confidence_score': 0.85,
                    'daily_forecasts': daily_forecasts
                },
                'pricing_strategy': pricing_strategy,
                'inventory_management': reorder_recommendation,
                'business_alerts': business_alerts,
                'overall_health_score': health_score
            }
            
            print("✅ Analysis completed successfully!")
            return report
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            # Return a basic error report
            return self.generate_error_report(input_data, str(e))
    
    def validate_input_data(self, input_data):
        """Validate and clean input data"""
        validated = input_data.copy()
        
        # Ensure numeric fields
        numeric_fields = ['Inventory Level', 'Units Ordered', 'Units Sold', 'Price', 
                         'Discount', 'Competitor Pricing', 'Demand Forecast']
        
        for field in numeric_fields:
            if field in validated:
                try:
                    validated[field] = float(validated[field])
                except (ValueError, TypeError):
                    validated[field] = 0.0
        
        # Set default values for missing fields
        defaults = {
            'Inventory Level': 100,
            'Units Ordered': 50,
            'Units Sold': 40,
            'Price': 30.0,
            'Discount': 10.0,
            'Competitor Pricing': 28.0,
            'Demand Forecast': 40.0
        }
        
        for field, default in defaults.items():
            if field not in validated or validated[field] == 0:
                validated[field] = default
        
        return validated
    
    def calculate_health_score(self, demand, reorder_recommendation):
        """Calculate overall product health score (0-100)"""
        score = 50
        
        # Demand component (0-40 points)
        if demand > 100:
            score += 30
        elif demand > 50:
            score += 20
        elif demand > 20:
            score += 10
        
        # Inventory component (0-30 points)
        urgency = reorder_recommendation['reorder_urgency']
        if urgency == "LOW":
            score += 20
        elif urgency == "MEDIUM":
            score += 10
        
        # Confidence component (0-20 points)
        score += 15  # Base confidence
        
        return min(100, max(0, round(score)))
    
    def generate_error_report(self, input_data, error_message):
        """Generate a basic report when analysis fails"""
        return {
            'product_info': {
                'store_id': input_data.get('Store ID', 'Unknown'),
                'product_id': input_data.get('Product ID', 'Unknown'),
                'category': input_data.get('Category', 'Unknown'),
                'region': input_data.get('Region', 'Unknown'),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'demand_forecast': {
                'immediate_demand': 50,
                'total_30day_forecast': 1500,
                'confidence_score': 0.5,
                'daily_forecasts': self.generate_dummy_forecast(50)
            },
            'pricing_strategy': {
                'current_price': input_data.get('Price', 30),
                'current_discount': input_data.get('Discount', 10),
                'recommended_price': input_data.get('Price', 30),
                'recommended_discount': input_data.get('Discount', 10),
                'pricing_strategy': 'MAINTAIN',
                'reasoning': 'Using current pricing due to analysis error',
                'expected_impact': {
                    'expected_volume_change': 0,
                    'expected_revenue_change': 0,
                    'old_revenue': 1500,
                    'new_revenue': 1500
                }
            },
            'inventory_management': {
                'current_inventory': input_data.get('Inventory Level', 100),
                'days_of_supply': 2.0,
                'reorder_point': 100,
                'recommended_order_quantity': 50,
                'reorder_urgency': 'MEDIUM',
                'reorder_action': 'MONITOR',
                'reasoning': 'Basic recommendation due to analysis error',
                'stockout_risk': 'Unknown risk due to analysis error'
            },
            'business_alerts': [{
                'type': 'SYSTEM_ERROR',
                'severity': 'HIGH',
                'message': 'Analysis encountered an error',
                'details': error_message,
                'action': 'Please try again or contact support'
            }],
            'overall_health_score': 50
        }