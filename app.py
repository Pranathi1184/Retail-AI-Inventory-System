from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import pickle
import os
from flask import make_response
from datetime import datetime, timedelta
from models import AdvancedRetailPredictor, db, User, PredictionHistory, InventoryAlert

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///retail.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Initialize predictor
predictor = AdvancedRetailPredictor()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))


@app.route('/export/json/<int:prediction_id>')
def export_json(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = PredictionHistory.query.filter_by(
        id=prediction_id,
        user_id=session['user_id']
    ).first_or_404()
    
    report = json.loads(prediction.prediction_result)
    
    # Create export data
    export_data = {
        'export_info': {
            'exported_at': datetime.now().isoformat(),
            'prediction_id': prediction_id,
            'product_id': prediction.product_id,
            'store_id': prediction.store_id
        },
        'product_info': report['product_info'],
        'demand_forecast': report['demand_forecast'],
        'pricing_strategy': report['pricing_strategy'],
        'inventory_management': report['inventory_management'],
        'business_alerts': report['business_alerts'],
        'overall_health_score': report['overall_health_score']
    }
    
    response = jsonify(export_data)
    response.headers['Content-Disposition'] = f'attachment; filename=retail_analysis_{prediction.product_id}_{prediction_id}.json'
    response.headers['Content-Type'] = 'application/json'
    
    return response

@app.route('/export/csv/<int:prediction_id>')
def export_csv(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = PredictionHistory.query.filter_by(
        id=prediction_id,
        user_id=session['user_id']
    ).first_or_404()
    
    report = json.loads(prediction.prediction_result)
    
    # Create CSV data
    csv_data = []
    
    # Basic info
    csv_data.append(['RETAIL ANALYSIS EXPORT'])
    csv_data.append(['Exported at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    csv_data.append(['Prediction ID', prediction_id])
    csv_data.append(['Product ID', prediction.product_id])
    csv_data.append(['Store ID', prediction.store_id])
    csv_data.append([])
    
    # Product Information
    csv_data.append(['PRODUCT INFORMATION'])
    csv_data.append(['Store', report['product_info']['store_id']])
    csv_data.append(['Product', report['product_info']['product_id']])
    csv_data.append(['Category', report['product_info']['category']])
    csv_data.append(['Region', report['product_info']['region']])
    csv_data.append(['Analysis Timestamp', report['product_info']['analysis_timestamp']])
    csv_data.append(['Health Score', report['overall_health_score']])
    csv_data.append([])
    
    # Demand Forecast
    csv_data.append(['DEMAND FORECAST'])
    csv_data.append(['Immediate Demand', report['demand_forecast']['immediate_demand']])
    csv_data.append(['30-Day Forecast', report['demand_forecast']['total_30day_forecast']])
    csv_data.append(['Confidence Score', report['demand_forecast']['confidence_score']])
    csv_data.append([])
    
    # Daily Forecast (first 7 days)
    csv_data.append(['DAILY FORECAST (Next 7 Days)'])
    csv_data.append(['Date', 'Forecasted Demand'])
    for i, (date, forecast) in enumerate(list(report['demand_forecast']['daily_forecasts'].items())[:7]):
        csv_data.append([date, forecast])
    csv_data.append([])
    
    # Pricing Strategy
    csv_data.append(['PRICING STRATEGY'])
    csv_data.append(['Current Price', report['pricing_strategy']['current_price']])
    csv_data.append(['Recommended Price', report['pricing_strategy']['recommended_price']])
    csv_data.append(['Current Discount', report['pricing_strategy']['current_discount']])
    csv_data.append(['Recommended Discount', report['pricing_strategy']['recommended_discount']])
    csv_data.append(['Strategy', report['pricing_strategy']['pricing_strategy']])
    csv_data.append(['Expected Revenue Change', report['pricing_strategy']['expected_impact']['expected_revenue_change']])
    csv_data.append([])
    
    # Inventory Management
    csv_data.append(['INVENTORY MANAGEMENT'])
    csv_data.append(['Current Inventory', report['inventory_management']['current_inventory']])
    csv_data.append(['Days of Supply', report['inventory_management']['days_of_supply']])
    csv_data.append(['Reorder Point', report['inventory_management']['reorder_point']])
    csv_data.append(['Recommended Order Quantity', report['inventory_management']['recommended_order_quantity']])
    csv_data.append(['Reorder Urgency', report['inventory_management']['reorder_urgency']])
    csv_data.append(['Stockout Risk', report['inventory_management']['stockout_risk']])
    csv_data.append([])
    
    # Business Alerts
    csv_data.append(['BUSINESS ALERTS'])
    if report['business_alerts']:
        csv_data.append(['Type', 'Severity', 'Message', 'Action'])
        for alert in report['business_alerts']:
            csv_data.append([
                alert['type'],
                alert['severity'],
                alert['message'],
                alert['action']
            ])
    else:
        csv_data.append(['No active alerts'])
    
    # Convert to CSV string
    csv_string = '\n'.join([','.join(map(str, row)) for row in csv_data])
    
    response = make_response(csv_string)
    response.headers['Content-Disposition'] = f'attachment; filename=retail_analysis_{prediction.product_id}_{prediction_id}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response

@app.route('/export/all/json')
def export_all_json():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    predictions = PredictionHistory.query.filter_by(user_id=user_id).all()
    
    export_data = {
        'export_info': {
            'exported_at': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'user_id': user_id,
            'username': session['username']
        },
        'predictions': []
    }
    
    for prediction in predictions:
        report = json.loads(prediction.prediction_result)
        prediction_data = {
            'prediction_id': prediction.id,
            'created_at': prediction.created_at.isoformat(),
            'store_id': prediction.store_id,
            'product_id': prediction.product_id,
            'category': prediction.category,
            'demand_forecast': prediction.demand_forecast,
            'recommended_price': prediction.recommended_price,
            'reorder_quantity': prediction.reorder_quantity,
            'health_score': report.get('overall_health_score', 0)
        }
        export_data['predictions'].append(prediction_data)
    
    response = jsonify(export_data)
    response.headers['Content-Disposition'] = f'attachment; filename=retail_analysis_all_{session["username"]}.json'
    response.headers['Content-Type'] = 'application/json'
    
    return response

@app.route('/export/all/csv')
def export_all_csv():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    predictions = PredictionHistory.query.filter_by(user_id=user_id).all()
    
    csv_data = []
    
    # Header
    csv_data.append(['COMPLETE PREDICTION HISTORY EXPORT'])
    csv_data.append(['Exported at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    csv_data.append(['User', session['username']])
    csv_data.append(['Total Predictions', len(predictions)])
    csv_data.append([])
    
    # Column headers
    csv_data.append([
        'Prediction ID',
        'Date',
        'Store ID',
        'Product ID',
        'Category',
        'Demand Forecast',
        'Recommended Price',
        'Reorder Quantity',
        'Health Score'
    ])
    
    # Data rows
    for prediction in predictions:
        report = json.loads(prediction.prediction_result)
        csv_data.append([
            prediction.id,
            prediction.created_at.strftime('%Y-%m-%d'),
            prediction.store_id,
            prediction.product_id,
            prediction.category,
            prediction.demand_forecast,
            prediction.recommended_price,
            prediction.reorder_quantity,
            report.get('overall_health_score', 0)
        ])
    
    # Convert to CSV string
    csv_string = '\n'.join([','.join(map(str, row)) for row in csv_data])
    
    response = make_response(csv_string)
    response.headers['Content-Disposition'] = f'attachment; filename=retail_analysis_all_{session["username"]}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get recent predictions
    recent_predictions = PredictionHistory.query.filter_by(
        user_id=user_id
    ).order_by(PredictionHistory.created_at.desc()).limit(5).all()
    
    # Get active alerts
    active_alerts = InventoryAlert.query.filter_by(
        user_id=user_id,
        resolved=False
    ).order_by(InventoryAlert.created_at.desc()).limit(5).all()
    
    # Sample analytics data
    analytics_data = {
        'total_predictions': PredictionHistory.query.filter_by(user_id=user_id).count(),
        'active_alerts': InventoryAlert.query.filter_by(user_id=user_id, resolved=False).count(),
        'high_demand_products': 12,  # This would come from actual data
        'reorder_recommendations': 8  # This would come from actual data
    }
    
    return render_template('dashboard.html',
                         recent_predictions=recent_predictions,
                         active_alerts=active_alerts,
                         analytics=analytics_data)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Collect form data
            product_data = {
                'Date': request.form.get('date') or datetime.now().strftime('%Y-%m-%d'),
                'Store ID': request.form.get('store_id'),
                'Product ID': request.form.get('product_id'),
                'Category': request.form.get('category'),
                'Region': request.form.get('region'),
                'Inventory Level': float(request.form.get('inventory_level', 0)),
                'Units Ordered': float(request.form.get('units_ordered', 0)),
                'Units Sold': float(request.form.get('units_sold', 0)),
                'Price': float(request.form.get('price', 0)),
                'Discount': float(request.form.get('discount', 0).replace('%', '')),
                'Competitor Pricing': float(request.form.get('competitor_pricing', 0)),
                'Weather Condition': request.form.get('weather_condition'),
                'Holiday/Promotion': 1 if request.form.get('holiday_promotion') in ['1', 'on', 'true'] else 0,
                'Seasonality': request.form.get('seasonality'),
                'Demand Forecast': float(request.form.get('demand_forecast', 0))
            }
            
            # Run prediction
            report = predictor.comprehensive_analysis(product_data)
            
            # Save prediction to database
            prediction_record = PredictionHistory(
                user_id=session['user_id'],
                store_id=product_data['Store ID'],
                product_id=product_data['Product ID'],
                category=product_data['Category'],
                input_data=json.dumps(product_data),
                prediction_result=json.dumps(report),
                demand_forecast=report['demand_forecast']['immediate_demand'],
                recommended_price=report['pricing_strategy']['recommended_price'],
                reorder_quantity=report['inventory_management']['recommended_order_quantity']
            )
            db.session.add(prediction_record)
            db.session.commit()
            
            # Create alerts if needed
            if report['inventory_management']['reorder_urgency'] in ['HIGH', 'MEDIUM']:
                alert = InventoryAlert(
                    user_id=session['user_id'],
                    prediction_id=prediction_record.id,
                    store_id=product_data['Store ID'],
                    product_id=product_data['Product ID'],
                    alert_type='REORDER',
                    severity=report['inventory_management']['reorder_urgency'],
                    message=f"Reorder needed for {product_data['Product ID']}",
                    details=report['inventory_management']['reasoning']
                )
                db.session.add(alert)
            
            for business_alert in report['business_alerts']:
                if business_alert['severity'] in ['HIGH', 'MEDIUM']:
                    alert = InventoryAlert(
                        user_id=session['user_id'],
                        prediction_id=prediction_record.id,
                        store_id=product_data['Store ID'],
                        product_id=product_data['Product ID'],
                        alert_type=business_alert['type'],
                        severity=business_alert['severity'],
                        message=business_alert['message'],
                        details=business_alert['details']
                    )
                    db.session.add(alert)
            
            db.session.commit()
            
            # Store report in session for results page
            session['last_prediction'] = report
            session['prediction_id'] = prediction_record.id
            
            return redirect(url_for('results'))
            
        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'error')
            return render_template('prediction.html')
    
    return render_template('prediction.html')

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    report = session.get('last_prediction')
    if not report:
        flash('No prediction results found!', 'error')
        return redirect(url_for('prediction'))
    
    return render_template('results.html', report=report)

@app.route('/analysis/<int:prediction_id>')
def analysis(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction_record = PredictionHistory.query.filter_by(
        id=prediction_id,
        user_id=session['user_id']
    ).first_or_404()
    
    report = json.loads(prediction_record.prediction_result)

    daily_forecasts = report["demand_forecast"]["daily_forecasts"]
    forecast_items = list(daily_forecasts.items())

    enhanced_forecasts = []
    for i, (date, value) in enumerate(forecast_items):
        prev_value = forecast_items[i - 1][1] if i > 0 else value
        trend = value - prev_value
        percent_change = ((value - prev_value) / prev_value * 100) if prev_value > 0 else 0


        enhanced_forecasts.append({
            "date": date,
            "value": value,
            "trend": trend,
            "percent_change": percent_change
        })


    return render_template('analysis.html', report=report, prediction_id=prediction_id)

@app.route('/api/demand_forecast', methods=['POST'])
def api_demand_forecast():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    product_data = data.get('product_data')
    
    try:
        report = predictor.comprehensive_analysis(product_data)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def api_alerts():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    alerts = InventoryAlert.query.filter_by(
        user_id=user_id,
        resolved=False
    ).order_by(InventoryAlert.created_at.desc()).limit(10).all()
    
    alerts_data = []
    for alert in alerts:
        alerts_data.append({
            'id': alert.id,
            'type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'details': alert.details,
            'created_at': alert.created_at.isoformat(),
            'store_id': alert.store_id,
            'product_id': alert.product_id
        })
    
    return jsonify(alerts_data)

@app.route('/api/resolve_alert/<int:alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    alert = InventoryAlert.query.filter_by(
        id=alert_id,
        user_id=session['user_id']
    ).first()
    
    if alert:
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True})
    
    return jsonify({'error': 'Alert not found'}), 404

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    predictions = PredictionHistory.query.filter_by(
        user_id=session['user_id']
    ).order_by(PredictionHistory.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('history.html', predictions=predictions)

@app.route('/prediction-type')
def prediction_type():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('prediction_type.html')

@app.route('/predict/bulk', methods=['GET', 'POST'])
def bulk_prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("No file uploaded", "error")
            return redirect(request.url)

        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            flash("Unsupported file format", "error")
            return redirect(request.url)

        predictor = AdvancedRetailPredictor()

        # CSV → internal feature mapping
        COLUMN_MAPPING = {
            "store_id": "Store ID",
            "product_id": "Product ID",
            "category": "Category",
            "price": "Price",
            "inventory_level": "Inventory Level",
            "units_sold": "Units Sold",
            "discount": "Discount",
            "competitor_price": "Competitor Pricing",
            "promotion": "Holiday/Promotion"
        }

        results = []

        for _, row in df.iterrows():
            raw = row.to_dict()

            # Map CSV columns to internal format
            mapped_input = {}
            for csv_col, internal_col in COLUMN_MAPPING.items():
                mapped_input[internal_col] = raw.get(csv_col)

            # Run full analysis
            report = predictor.comprehensive_analysis(mapped_input)

            # 🔥 SAVE TO DATABASE (THIS IS THE KEY FIX)
            history = PredictionHistory(
                user_id=session['user_id'],
                store_id=report["product_info"]["store_id"],
                product_id=report["product_info"]["product_id"],
                category=report["product_info"]["category"],
                input_data=json.dumps(mapped_input),
                demand_forecast=report["demand_forecast"]["immediate_demand"],
                recommended_price=report["pricing_strategy"]["recommended_price"],
                reorder_quantity=int(
                report["inventory_management"].get("recommended_order_quantity", 0)
                ),
                prediction_result=json.dumps(report),
                created_at=datetime.utcnow()
            )

            db.session.add(history)
            db.session.flush()  # 🔥 get history.id before commit

            # Prepare UI results
            results.append({
                "id": history.id,
                "store_id": report["product_info"]["store_id"],
                "product_id": report["product_info"]["product_id"],
                "category": report["product_info"]["category"],
                "predicted_demand": round(report["demand_forecast"]["immediate_demand"], 2),
                "reorder_qty": report["inventory_management"]["recommended_order_quantity"],
                "recommended_price": report["pricing_strategy"]["recommended_price"],
                "health_score": report["overall_health_score"]
            })

        # Commit once (best practice)
        db.session.commit()

        return render_template("bulk_results.html", results=results)

    return render_template("bulk_prediction.html")



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)