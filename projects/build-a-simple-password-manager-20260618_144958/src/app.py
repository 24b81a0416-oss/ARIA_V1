from flask import Flask, request, jsonify
from src.authentication import Authentication
from src.password_generator import PasswordGenerator
from src.password_vault import PasswordVault

app = Flask(__name__)

authentication = Authentication()
password_generator = PasswordGenerator()
password_vault = PasswordVault()

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    authentication.register(username, password)
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    """Login a user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if authentication.login(username, password):
        return jsonify({'message': 'User logged in successfully'}), 200
    return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/generate_password', methods=['POST'])
def generate_password():
    """Generate a password"""
    data = request.json
    length = data.get('length', 12)
    password = password_generator.generate_password(length)
    return jsonify({'password': password}), 200

@app.route('/store_password', methods=['POST'])
def store_password():
    """Store a password"""
    data = request.json
    user_id = data.get('user_id')
    service = data.get('service')
    password = data.get('password')
    password_vault.store_password(user_id, service, password)
    return jsonify({'message': 'Password stored successfully'}), 201

@app.route('/retrieve_password', methods=['POST'])
def retrieve_password():
    """Retrieve a password"""
    data = request.json
    user_id = data.get('user_id')
    service = data.get('service')
    password = password_vault.retrieve_password(user_id, service)
    if password:
        return jsonify({'password': password}), 200
    return jsonify({'message': 'Password not found'}), 404