import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')
import pickle
import json
from datetime import datetime
from sklearn.linear_model import Ridge


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


class RetailDemandForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.results = {}
        self.feature_importance = {}

    """def inject_controlled_noise(self, df):
    
        Inject realistic retail noise into Units Sold
        

        rng = np.random.default_rng(42)

        # --- 1. Multiplicative demand noise (day-to-day volatility)
        noise = rng.normal(loc=1.0, scale=0.15, size=len(df))
        df['Units Sold'] = df['Units Sold'] * noise

        # --- 2. Promotion failure noise (discount doesn't always work)
        promo_mask = df['Discount'] > 0
        promo_noise = rng.uniform(0.7, 1.2, promo_mask.sum())
        df.loc[promo_mask, 'Units Sold'] *= promo_noise

        # --- 3. Random demand shocks (rare events)
        shock_idx = rng.choice(
            df.index,
            size=int(0.02 * len(df)),  # 2% of days
            replace=False
        )
        df.loc[shock_idx, 'Units Sold'] *= rng.uniform(1.3, 2.0, len(shock_idx))

        # --- 4. Clip & round
        df['Units Sold'] = (
            df['Units Sold']
            .clip(lower=0)
            .round()
            .astype(int)
        )

        return df"""

            
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the retail data"""
        print("Loading data...")
        self.df = pd.read_csv(file_path)

        #self.df = self.inject_controlled_noise(self.df)
        
        # Convert date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by date
        self.df = self.df.sort_values(['Date', 'Store ID', 'Product ID'])
        
        # Create time-based features
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek
        self.df['day_of_month'] = self.df['Date'].dt.day
        self.df['month'] = self.df['Date'].dt.month
        self.df['quarter'] = self.df['Date'].dt.quarter
        self.df['year'] = self.df['Date'].dt.year
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def create_lag_features(self, df, lag_days=[1, 2, 3, 7, 14, 28]):
        """Create lag features for time series data"""
        print("Creating lag features...")
        
        # Sort by date and create unique store-product identifier
        df = df.sort_values(['Store ID', 'Product ID', 'Date'])
        df['store_product'] = df['Store ID'] + '_' + df['Product ID']
        
        # Create lag features for units sold
        for lag in lag_days:
            df[f'units_sold_lag_{lag}'] = df.groupby('store_product')['Units Sold'].shift(lag)
            df[f'inventory_lag_{lag}'] = df.groupby('store_product')['Inventory Level'].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            df[f'units_sold_rolling_mean_{window}'] = df.groupby('store_product')['Units Sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'units_sold_rolling_std_{window}'] = df.groupby('store_product')['Units Sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'units_sold_rolling_median_{window}'] = df.groupby('store_product')['Units Sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).median()
            )

        
        # Fill NaN values created by rolling/lag features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
        
        # Drop the temporary column
        df = df.drop('store_product', axis=1)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_columns = ['Store ID', 'Product ID', 'Category', 'Region', 
                             'Weather Condition', 'Seasonality']
        
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def prepare_features(self, target='Units Sold'):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Create lag features
        df_with_lags = self.create_lag_features(self.df.copy())
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_with_lags)
        
        # Define feature columns
        feature_columns = [
            'Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level',
            'Units Ordered', 'Price', 'Discount', 
            'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality',
            'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_weekend'
        ]
        
        # Add lag features
        lag_features = [col for col in df_encoded.columns if 'lag_' in col or 'rolling_' in col]
        feature_columns.extend(lag_features)
        
        # Remove any features that might not exist
        feature_columns = [col for col in feature_columns if col in df_encoded.columns]
        
        # Prepare X and y
        X = df_encoded[feature_columns]
        MAX_DEMAND = 1000   # <-- add here (or make it class-level constant)

        y = np.log1p(
            np.clip(df_encoded[target], 0, MAX_DEMAND)
        )


        
        # Remove rows with NaN values (from lag features)
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Target variable stats - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        
        return X, y, feature_columns
    
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """Train individual models"""
        print("\nTraining individual models...")
        # ---- Create separate copies for CatBoost ----
        X_train_cb = X_train.copy()
        X_val_cb = X_val.copy()
        self.X_val_cb = X_val_cb
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # LightGBM - Clean feature names
        print("Training LightGBM...")
        X_train_clean = X_train.copy()
        X_train_clean.columns = [col.replace(' ', '_').replace('-', '_') for col in X_train_clean.columns]
        X_val_clean = X_val.copy()
        X_val_clean.columns = [col.replace(' ', '_').replace('-', '_') for col in X_val_clean.columns]
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            num_leaves=128,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        lgb_model.fit(X_train_clean, y_train)
        self.models['lightgbm'] = lgb_model
        self.X_val_clean = X_val_clean  # Store for prediction
        
        # CatBoost
        print("Training CatBoost...")
        cat_cols = [
            'Store ID', 'Product ID', 'Category', 'Region',
            'Weather Condition', 'Seasonality', 'Holiday/Promotion'
        ]

        for col in cat_cols:
            if col in X_train.columns:
                 X_train_cb[col] = X_train_cb[col].astype(str)
                 X_val_cb[col] = X_val_cb[col].astype(str)

        cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols if c in X_train_cb.columns]

        cat_model = cb.CatBoostRegressor(
            iterations=800,
            depth=10,
            learning_rate=0.05,
            loss_function='RMSE',
            cat_features=cat_idx,
            random_state=42,
            verbose=False
        )

        cat_model.fit(X_train_cb, y_train)

        self.models['catboost'] = cat_model
        
        # Evaluate individual models
        self.evaluate_individual_models(X_val, y_val)
    
    def evaluate_individual_models(self, X_val, y_val):
        """Evaluate individual models on validation set"""

        print("\nEvaluating individual models...")

        y_val_inv = np.expm1(y_val)

        for name, model in self.models.items():

            if name == 'lightgbm':
                y_pred = model.predict(self.X_val_clean)
            elif name == 'catboost':
                y_pred = model.predict(self.X_val_cb)
            else:
                y_pred = model.predict(X_val)

            y_pred_inv = np.expm1(y_pred)

            # ---- Metrics ----
            r2 = r2_score(y_val_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))
            mae = mean_absolute_error(y_val_inv, y_pred_inv)

            # ---- MAPE (safe) ----
            eps = 1e-6
            mape = np.mean(
                np.abs((y_val_inv - y_pred_inv) / np.maximum(y_val_inv, eps))
            ) * 100

            # ---- WAPE (better for demand) ----
            wape = np.sum(np.abs(y_val_inv - y_pred_inv)) / np.sum(y_val_inv) * 100

            # ✅ CREATE entry first
            self.results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'wape': wape,
                'predictions': y_pred_inv
            }

            print(
                f"{name:15} | "
                f"R²: {r2:.4f} | "
                f"RMSE: {rmse:.2f} | "
                f"MAPE: {mape:.2f}% | "
                f"WAPE: {wape:.2f}%"
            )

    
    """def create_hybrid_model(self, X_train, y_train, X_val, y_val):
        Create hybrid model using weighted ensemble
        print("\nCreating hybrid model...")
        
        # Get predictions from all models
        predictions = {}
        weights = {}
        
        for name, model in self.models.items():
            if name == 'lightgbm':
                # Use cleaned features for LightGBM
                pred = self.results[name]['predictions']
            else:
                pred = self.results[name]['predictions']
            
            predictions[name] = pred
            
            # Calculate weight based on R² score
            r2 = self.results[name]['r2']
            weights[name] = max(0, r2)  # Ensure non-negative weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight
        
        print("Model weights for hybrid:")
        for name, weight in weights.items():
            print(f"{name:15} - Weight: {weight:.4f}")
        
        # Create hybrid predictions - ensure float dtype
        hybrid_pred = np.zeros(len(y_val), dtype=np.float64)
        for name, pred in predictions.items():
            hybrid_pred += weights[name] * pred.astype(np.float64)
        
        y_val_inv = np.expm1(y_val)
        # Calculate hybrid model metrics
        hybrid_r2 = r2_score(y_val_inv, hybrid_pred)
        hybrid_rmse = np.sqrt(mean_squared_error(y_val_inv, hybrid_pred))
        hybrid_mae = mean_absolute_error(y_val_inv, hybrid_pred)
        
        self.results['hybrid'] = {
            'r2': hybrid_r2,
            'rmse': hybrid_rmse,
            'mae': hybrid_mae,
            'weights': weights,
            'predictions': hybrid_pred
        }
        
        print(f"Hybrid Model    - R²: {hybrid_r2:.4f}, RMSE: {hybrid_rmse:.4f}, MAE: {hybrid_mae:.4f}")
        
        return weights"""
        
    def create_hybrid_model(self, X_train, y_train, X_val, y_val):
        """Create hybrid model using STACKING"""
        print("\nCreating hybrid model (STACKING)...")

        # Base model predictions (already inverse-log)
        rf_pred = self.results['random_forest']['predictions']
        xgb_pred = self.results['xgboost']['predictions']
        lgb_pred = self.results['lightgbm']['predictions']
        cat_pred = self.results['catboost']['predictions']

        meta_X = np.column_stack([rf_pred, xgb_pred, lgb_pred, cat_pred])

        # True values
        y_val_inv = np.expm1(y_val)

        # Stack predictions
        split = int(0.5 * len(meta_X))
        meta_X_train, meta_X_test = meta_X[:split], meta_X[split:]
        y_meta_train, y_meta_test = y_val_inv[:split], y_val_inv[split:]

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, y_meta_train)

        hybrid_pred = meta_model.predict(meta_X)

        # Metrics
        hybrid_r2 = r2_score(y_val_inv, hybrid_pred)
        hybrid_rmse = np.sqrt(mean_squared_error(y_val_inv, hybrid_pred))
        hybrid_mae = mean_absolute_error(y_val_inv, hybrid_pred)

        eps = 1e-6
        hybrid_mape = np.mean(
            np.abs((y_val_inv - hybrid_pred) / np.maximum(y_val_inv, eps))
        ) * 100

        hybrid_wape = np.sum(np.abs(y_val_inv - hybrid_pred)) / np.sum(y_val_inv) * 100

        self.results['hybrid'] = {
            'r2': hybrid_r2,
            'rmse': hybrid_rmse,
            'mae': hybrid_mae,
            'mape': hybrid_mape,
            'wape': hybrid_wape,
            'predictions': hybrid_pred
        }

        print(
            f"Hybrid Model | "
            f"R²: {hybrid_r2:.4f} | "
            f"RMSE: {hybrid_rmse:.2f} | "
            f"MAPE: {hybrid_mape:.2f}% | "
            f"WAPE: {hybrid_wape:.2f}%"
        )
    
    def plot_results(self, X_val, y_val):
        """Plot accuracy graphs and results"""
        print("\nGenerating plots...")
        
        # 1. Model Comparison Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model performance comparison
        models_list = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models_list]
        rmse_scores = [self.results[model]['rmse'] for model in models_list]
        
        # R² scores
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple']
        bars1 = axes[0, 0].bar(models_list, r2_scores, color=colors[:len(models_list)])
        axes[0, 0].set_title('Model R² Scores Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, v in zip(bars1, r2_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE scores
        bars2 = axes[0, 1].bar(models_list, rmse_scores, color=colors[:len(models_list)])
        axes[0, 1].set_title('Model RMSE Scores Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, v in zip(bars2, rmse_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Actual vs Predicted for Hybrid Model
        hybrid_pred = self.results['hybrid']['predictions']
        y_val = np.expm1(y_val)

        # Clip extreme values for stable visualization
        upper = np.percentile(y_val, 99)
        hybrid_pred_plot = np.clip(hybrid_pred, 0, upper)
        y_val_plot = np.clip(y_val, 0, upper)


        axes[1, 0].scatter(y_val_plot, hybrid_pred_plot, alpha=0.6, s=10, color='blue')
        max_val = max(y_val_plot.max(), hybrid_pred_plot.max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.8)
        axes[1, 0].set_xlabel('Actual Values', fontsize=12)
        axes[1, 0].set_ylabel('Predicted Values', fontsize=12)
        axes[1, 0].set_title('Hybrid Model: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add R² to plot
        hybrid_r2 = self.results['hybrid']['r2']
        axes[1, 0].text(0.05, 0.95, f'R² = {hybrid_r2:.4f}', 
                       transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 3. Feature Importance (using Random Forest)
        if 'random_forest' in self.models:
            feature_importance = self.models['random_forest'].feature_importances_
            feature_names = X_val.columns
            
            # Get top 15 features
            indices = np.argsort(feature_importance)[-15:]
            
            axes[1, 1].barh(range(len(indices)), feature_importance[indices], color='lightblue', alpha=0.8)
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
            axes[1, 1].set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Feature Importance', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Residual Plot for Hybrid Model
        plt.figure(figsize=(10, 6))
        residuals = y_val_plot.values - hybrid_pred_plot
        plt.scatter(hybrid_pred_plot, residuals, alpha=0.6, s=10, color='green')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Hybrid Model: Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('hybrid_model_residuals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Time Series of Predictions (sample)
        plt.figure(figsize=(12, 6))
        sample_size = min(200, len(y_val))
        plt.plot(range(sample_size), y_val.values[:sample_size], 'b-', label='Actual', alpha=0.8, linewidth=1.5)
        plt.plot(range(sample_size), hybrid_pred[:sample_size], 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Units Sold', fontsize=12)
        plt.title('Sample Time Series: Actual vs Predicted (Hybrid Model)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('time_series_prediction_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        
        # 7. Prediction Distribution

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(y_val_plot, bins=50, alpha=0.7, color='blue', label='Actual')
        plt.xlabel('Units Sold')
        plt.ylabel('Frequency')
        plt.title('Actual Values Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(hybrid_pred_plot, bins=50, alpha=0.7, color='red', label='Predicted')
        plt.xlabel('Units Sold')
        plt.ylabel('Frequency')
        plt.title('Predicted Values Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\nSaving models...")
        
        # Save models
        for name, model in self.models.items():
            with open(f'{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save preprocessing objects
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'results': self.results
            }, f)
        
        # Save results as JSON
        results_json = {}
        for model_name, metrics in self.results.items():
            if 'weights' in metrics:
                results_json[model_name] = {
                    'r2': float(metrics['r2']),
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'weights': {k: float(v) for k, v in metrics['weights'].items()}
                }
            else:
                results_json[model_name] = {
                    'r2': float(metrics['r2']),
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae'])
                }
        
        with open('model_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print("Models and results saved successfully!")
    
    def train_complete_pipeline(self, file_path):
        print("Starting Retail Demand Forecasting Training Pipeline")
        print("=" * 50)

        # Load augmented data
        self.load_and_preprocess_data(file_path)

        # Prepare features
        X, y, feature_columns = self.prepare_features()

        # Time-based split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Reset containers
        self.models = {}
        self.results = {}

        # Train base models
        self.train_individual_models(X_train, y_train, X_val, y_val)

        # Stacked hybrid
        self.create_hybrid_model(X_train, y_train, X_val, y_val)

        # Plots
        self.plot_results(X_val, y_val)

        # Save models
        self.save_models()

        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY")
        print("=" * 50)

        for model_name, metrics in self.results.items():
            print(f"{model_name:15} | R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")

        return self.results



       
# Main execution
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = RetailDemandForecaster()
    
    # Train complete pipeline
    results = forecaster.train_complete_pipeline(
    'retail_store_inventory_augmented.csv'
    )

    
    