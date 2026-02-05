#os.environ["GROQ_API_KEY"] = "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"

import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

app = Flask(__name__)
plt.switch_backend('Agg')

# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"  # Replace with your actual key
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
DATA_DIR = "data"


# --- 1. DATA LOADING & CLEANING (EDA) ---
def load_and_clean_data():
    d_df = pd.read_csv(os.path.join(DATA_DIR, 'Expanded_Destinations.csv'))
    r_df = pd.read_csv(os.path.join(DATA_DIR, 'Final_Updated_Expanded_Reviews.csv'))
    h_df = pd.read_csv(os.path.join(DATA_DIR, 'Final_Updated_Expanded_UserHistory.csv'))
    u_df = pd.read_csv(os.path.join(DATA_DIR, 'Final_Updated_Expanded_Users.csv'))

    report = {
        'dest_dupes': d_df.duplicated(subset=['Name']).sum(),
        'hist_dupes': h_df.duplicated().sum(),
        'nulls': d_df.isnull().sum().sum() + u_df.isnull().sum().sum()
    }

    d_df = d_df.drop_duplicates(subset=['Name']).reset_index(drop=True)
    h_df = h_df.drop_duplicates().reset_index(drop=True)
    d_df = d_df.fillna("Not Specified")
    u_df = u_df.fillna("Not Specified")

    return d_df, r_df, h_df, u_df, report


df_dest, df_rev, df_hist, df_users, clean_report = load_and_clean_data()


# --- 2. MULTI-AGENT SYSTEM (LANGGRAPH) ---
class TravelState(TypedDict):
    user_id: int
    profile: dict
    history: str
    sentiment_data: str
    user_override: str
    prediction: str


def profiler_node(state):
    """Agent 1: Extracts User Profile and Travel History"""
    uid = int(state['user_id'])
    user = df_users[df_users['UserID'] == uid].iloc[0].to_dict()
    hist_data = df_hist[df_hist['UserID'] == uid].merge(df_dest, on='DestinationID')
    hist_summary = hist_data[['Name', 'ExperienceRating']].to_string(index=False)
    return {**state, "profile": user, "history": hist_summary}


def reviewer_node(state):
    """Agent 2: Analyzes Public Sentiment and Ratings"""
    pref_type = state['profile']['Preferences'].split(',')[0]
    # Fetch top 10 potential candidates to give the predictor enough options
    sample_dests = df_dest[df_dest['Type'] == pref_type].head(10)

    review_info = ""
    for _, row in sample_dests.iterrows():
        avg_rating = df_rev[df_rev['DestinationID'] == row['DestinationID']]['Rating'].mean()
        rating_text = f"{avg_rating:.1f}/5" if not pd.isna(avg_rating) else "4.4/5 (Highly Rated)"
        review_info += f"{row['Name']} ({row['State']}): {rating_text}. "

    return {**state, "sentiment_data": review_info}


def predictor_node(state):
    """Agent 3: Final Recommendation Engine"""
    user = state['profile']
    override = state.get('user_override', 'No special requests')

    prompt = f"""
    Traveler Name: {user['Name']}
    Interests: {user['Preferences']}
    Special Custom Request (What-If): {override}

    Past History: {state['history']}
    Public Vibe/Sentiment: {state['sentiment_data']}

    TASK: Provide exactly 5 destination recommendations. 
    You MUST follow this structure:
    1. A Markdown table with exactly 5 rows: | Rank | Destination | State | Type | Public Rating |
    2. A section titled "### Why these match {user['Name']}'s preferences:" 
       In this section, explain how the 5 choices align with their history AND their special request: "{override}".
    3. A section "### Destination Highlights" with 3 sentences for each of the 5 places.
    Be helpful, personal, and ensure you provide 5 options.
    """
    response = llm.invoke(prompt)
    return {**state, "prediction": response.content}


# Create Graph
builder = StateGraph(TravelState)
builder.add_node("profiler", profiler_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("predictor", predictor_node)

builder.set_entry_point("profiler")
builder.add_edge("profiler", "reviewer")
builder.add_edge("reviewer", "predictor")
builder.add_edge("predictor", END)
graph = builder.compile()


# --- 3. FLASK ROUTES ---
@app.route('/')
def index():
    uids = df_users['UserID'].unique().tolist()[:50]
    return render_template('index.html', user_ids=uids)


@app.route('/predict', methods=['POST'])
def predict():
    uid = request.form.get('user_id')
    note = request.form.get('custom_note', "")
    result = graph.invoke({"user_id": uid, "user_override": note})
    return jsonify({"name": result["profile"]["Name"], "recommendation": result["prediction"]})


@app.route('/eda')
def eda():
    # Chart 1
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_dest, x='Type', palette='viridis')
    plt.title('Destination Types')
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    chart1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # Chart 2
    plt.figure(figsize=(10, 5))
    sns.histplot(df_dest['Popularity'], kde=True, color='purple')
    plt.title('Popularity Distribution')
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    chart2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    return render_template('eda.html', report=clean_report, chart1=chart1, chart2=chart2)


if __name__ == '__main__':
    app.run(debug=True)