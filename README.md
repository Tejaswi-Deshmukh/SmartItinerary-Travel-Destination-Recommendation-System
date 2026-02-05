✈️ AI-Powered Travel Recommendation System

An intelligent travel recommendation web application built using Flask, LangGraph, and LLMs (LLaMA-3 via Groq) that provides personalized destination suggestions based on:

User profile

Travel history

Public sentiment & ratings

Custom user requests (What-If scenario)

The system follows a multi-agent architecture to generate meaningful and explainable travel recommendations.

🚀 Features

✔ Personalized destination recommendations
✔ Multi-agent decision pipeline using LangGraph
✔ Integration with LLM (LLaMA-3-70B via Groq)
✔ Public review & sentiment-based filtering
✔ Custom user override request support
✔ Interactive EDA dashboard with charts
✔ Clean and modular Flask web interface

🧠 System Architecture

The recommendation pipeline uses three intelligent agents:

1️⃣ Profiler Agent

Extracts user profile & preferences

Collects past travel history

2️⃣ Reviewer Agent

Analyzes public ratings & sentiment

Filters top candidate destinations

3️⃣ Predictor Agent

Uses LLM reasoning to generate:

Ranked recommendation table

Personalized explanation

Destination highlights

🛠️ Tech Stack

Backend: Flask, Python

AI/LLM: LangGraph, LangChain, Groq (LLaMA-3-70B)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Frontend: HTML, Jinja Templates

Architecture: Multi-Agent Workflow

📂 Project Structure
├── app.py
├── data/
│   ├── Expanded_Destinations.csv
│   ├── Final_Updated_Expanded_Reviews.csv
│   ├── Final_Updated_Expanded_UserHistory.csv
│   └── Final_Updated_Expanded_Users.csv
├── templates/
│   ├── index.html
│   └── eda.html
└── README.md

📊 Exploratory Data Analysis (EDA)

The /eda route provides:

Destination type distribution

Popularity distribution

Data cleaning report (duplicates & nulls)

This helps understand travel trends and dataset quality.

👨‍💻 Author

Tejaswi Deshmukh
