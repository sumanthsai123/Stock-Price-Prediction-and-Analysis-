from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import matplotlib.dates as mdates  
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sqlite3
    
app = Flask(__name__)
app.secret_key = 'Stock-market-prediction-24'  

USER_CREDENTIALS = {
    "admin": {"password": "password123", "email": "admin@example.com", "role": "admin"},
    "user1": {"password": "userpass", "email": "user@example.com", "role": "user"}
}

# Load pre-trained model
model = load_model("stock_future_prediction_saved.keras")

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_username = request.form.get("new_username")
        new_email = request.form.get("new_email")
        new_password = request.form.get("new_password")

        if not new_username or not new_email or not new_password:
            flash("All fields are required.", "error")
            return render_template("login.html", show_register=True)

        if new_username in USER_CREDENTIALS:
            flash("Username already exists.", "error")
            return render_template("login.html", show_register=True)

        # Assign "user" role to new users
        USER_CREDENTIALS[new_username] = {"password": new_password, "email": new_email, "role": "user"}

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("login.html", show_register=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username]["password"] == password:
            session["username"] = username
            session["role"] = USER_CREDENTIALS[username]["role"]  # Store role in session

            if session["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("index"))

        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")
@app.route("/admin_dashboard")
def admin_dashboard():
    if "username" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))

    with sqlite3.connect("feedback.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, feedback FROM feedback")
        feedback_list = cursor.fetchall()

    return render_template("admin_dashboard.html", feedback_list=feedback_list, users=USER_CREDENTIALS)

@app.route("/logout")
def logout():
    session.pop("username", None)  # Remove user from session
    return redirect(url_for("login"))
@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")
from functools import wraps

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "username" not in session or session.get("role") != role:
                flash("Unauthorized access!", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/index", methods=["GET", "POST"])
@role_required("user")  # Only normal users can access
def index():
    if request.method == "POST":
        stock = request.form.get("stock_id", "").strip().upper()
        if not stock:
            flash("Please enter a valid stock ID.", "error")
            return redirect(url_for("index"))

        if "," in stock or len(stock.split()) > 1:
            flash("Please enter a single ticker symbol, not comma-separated or multiple symbols.", "error")
            return redirect(url_for("index"))

        session["stock_id"] = stock
        return redirect(url_for("results"))

    return render_template("index.html")

@app.route("/results")
def results():
    if "username" not in session:  # Restrict access if not logged in
        return redirect(url_for("login"))

    stock = session.get("stock_id")
    if not stock:
        return redirect(url_for("index"))

    try:
        from datetime import datetime
        end = datetime.now()
        start = datetime(end.year - 15, end.month, end.day)

        mutual_fund_suffixes = ["X"]  # Most mutual funds end with 'X'
        stock_suffixes = [".O", ".N", ".K", ".Q"]  # Common US stock suffixes
 
        stock_upper = stock.upper()  # Convert to uppercase for uniformity

        if stock_upper.endswith("X") or "MF" in stock_upper:
          entity_type = "Mutual Fund"
        elif any(stock_upper.endswith(suffix) for suffix in stock_suffixes) or stock_upper.isalpha():
          entity_type = "Stock"
        else:
           entity_type = "Unknown"


        df = yf.download(stock, start, end)
        print(f"Fetched Data for {stock} ({entity_type}): {df.tail()}")

        if df.empty:
            return render_template("results.html", error_message=f"No data found for {entity_type} '{stock}'.")

        if 'Close' not in df.columns:
            return render_template("results.html", error_message=f"No Close price data available for '{stock}'.")

        Close_price = df['Close']
        if isinstance(Close_price, pd.DataFrame):
            if Close_price.shape[1] > 1:
                return render_template("results.html", error_message="Multiple tickers were detected. Please enter a single ticker symbol.")
            Close_price = Close_price.iloc[:, 0]

        Close_price = Close_price.dropna()
        if len(Close_price) < 101:
            return render_template("results.html", error_message=f"Not enough historical data for '{stock}'. Please try a different ticker.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(Close_price.values.reshape(-1, 1))
        print(f"Scaled Data for {stock}: {scaled_data[-5:]}")

        x_data = [scaled_data[i - 100:i] for i in range(100, len(scaled_data))]
        x_data = np.asarray(x_data, dtype=np.float32)
        if x_data.ndim != 3 or x_data.shape[1:] != (100, 1):
            return render_template("results.html", error_message="Failed to prepare model input data correctly. Please try a different ticker.")

        predicted_scaled = model.predict(x_data)
        predicted_prices = scaler.inverse_transform(predicted_scaled)
        predicted_prices = predicted_prices.flatten()
        print(f"Predicted Prices for {stock}: {predicted_prices[-5:]}")

        # Actual vs Predicted Prices Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index[-len(predicted_prices):], Close_price[-len(predicted_prices):], label="Actual Prices", color="blue")
        ax.plot(df.index[-len(predicted_prices):], predicted_prices, label="Model Predictions", color="orange")
        ax.set_title(f"{stock} ({entity_type}) - Actual vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        actual_vs_predicted_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Next 10 Days Prediction
        last_100_days = scaled_data[-100:].reshape(1, 100, 1)
        prediction_10_days = []

        for _ in range(10):
            next_day_pred = model.predict(last_100_days)
            next_day_value = float(next_day_pred[0, 0])
            prediction_10_days.append(next_day_value)
            next_100 = np.array(next_day_value, dtype=np.float32).reshape(1, 1, 1)
            last_100_days = np.concatenate([last_100_days[:, 1:, :], next_100], axis=1)

        prediction_10_days = np.array(prediction_10_days, dtype=np.float32).reshape(-1, 1)
        prediction_10_days = scaler.inverse_transform(prediction_10_days)
        prediction_10_days = prediction_10_days.flatten()

        last_date = df.index[-1]
        next_10_days = pd.date_range(last_date + pd.DateOffset(days=1), periods=10)
        predictions = [{"Date": date.date(), "Predicted": float(pred)} for date, pred in zip(next_10_days, prediction_10_days)]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Ensure the next day's predicted price is plotted as a line instead of a single dot
        ax.plot([df.index[-1], next_10_days[0]], [Close_price.iloc[-1], prediction_10_days[0]], 
                marker='o', linestyle='-', color='red', label="Next Day Prediction")

        ax.set_title(f"{stock} Next Day Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        plt.grid(True)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=30)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        next_day_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Next 10 Days Prediction Graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_10_days, prediction_10_days, label="Predicted Prices", color="purple")
        ax.set_title(f"{stock} Predicted Prices for Next 10 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        plt.xticks(rotation=30)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        next_10_days_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Closing Price Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price, label="Closing Prices", color="blue")
        ax.set_title(f"{stock} Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        closing_prices_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

          # Moving Averages Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price.rolling(window=100).mean(), label="100-Day MA", color="orange")
        ax.plot(df.index, Close_price.rolling(window=200).mean(), label="200-Day MA", color="green")
        ax.set_title(f"{stock} Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        moving_averages_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return render_template(
            "results.html",
            entity_type=entity_type, 
            predictions=predictions,
            plots={
                "actual_vs_predicted": actual_vs_predicted_plot,
                "next_1_day": next_day_plot,
                "next_10_days": next_10_days_plot,
                "closing_prices": closing_prices_plot,
                "moving_averages": moving_averages_plot
            }
        )
    

    except Exception as e:
        return render_template("results.html", error_message=f"An error occurred: {e}")
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    if "username" not in session:
        flash("You must be logged in to submit feedback.", "error")
        return redirect(url_for("login"))

    feedback = request.form.get("feedback")
    username = session["username"]

    with sqlite3.connect("feedback.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (username, feedback) VALUES (?, ?)", (username, feedback))
        conn.commit()

    flash("Feedback submitted successfully!", "success")
    return redirect(url_for("index"))
@app.route("/back_to_index")
def back_to_index():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

