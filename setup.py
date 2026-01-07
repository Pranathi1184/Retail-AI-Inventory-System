#!/usr/bin/env python3
"""
Setup script for RetailAI Prediction System
"""

import os
import sys
from app import app, db
from models import User
from werkzeug.security import generate_password_hash

def setup_application():
    """Initialize the application database and create admin user"""
    
    print("🚀 Setting up RetailAI Prediction System...")
    
    # Create database tables
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")
        
        # Create demo user
        if not User.query.filter_by(username='demo').first():
            demo_user = User(
                username='demo',
                email='demo@retailai.com',
                password=generate_password_hash('demo123')
            )
            db.session.add(demo_user)
            db.session.commit()
            print("✅ Demo user created (username: demo, password: demo123)")
        
        print("✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Login with: demo / demo123")

if __name__ == '__main__':
    setup_application()