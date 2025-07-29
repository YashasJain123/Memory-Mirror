import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import random

DATA_FILE = "data.json"

# Load existing journal data
def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

# Save journal data
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Sentiment analysis stub
def analyze_sentiment(text):
    if "sad" in text.lower() or "bad" in text.lower():
        return "NEGATIVE"
    elif "happy" in text.lower() or "grateful" in text.lower():
        return "POSITIVE"
    else:
        return "NEUTRAL"

# Login and user setup
data = load_data()
st.set_page_config(page_title="Memory Mirror", page_icon="🧠", layout="centered")
st.title("🧠 Memory Mirror - AI-Powered Journal")

# Authentication
email = st.text_input("Enter your email")
password = st.text_input("Enter a password", type="password")
if st.button("Login"):
    if email not in data:
        data[email] = {"password": password, "name": "", "entries": []}
        save_data(data)
    elif data[email]["password"] != password:
        st.error("Wrong password")
        st.stop()
    st.session_state["user"] = email

if "user" not in st.session_state:
    st.stop()

user_email = st.session_state["user"]
user = data[user_email]
if not user["name"]:
    name = st.text_input("What name do you want to display?")
    if name:
        user["name"] = name
        save_data(data)

# Navigation
page = st.sidebar.selectbox("Go to", ["📝 New Entry", "📚 My Journals", "📊 Graph", "🧠 Insights"])

# --- New Entry ---
if page == "📝 New Entry":
    st.subheader("📝 Write Your Journal Entry")
    st.markdown("*Tip: If you like to write on paper, use Google Lens/Camera to copy text and paste it here.*")

    journal = st.text_area("Your thoughts today", height=200)
    if st.button("Reflect"):
        if len(journal.split()) < 10:
            st.warning("Please write at least 10 words for a proper reflection.")
        else:
            sentiment = analyze_sentiment(journal)
            entry = {
                "text": journal,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "sentiment": sentiment
            }
            user["entries"].append(entry)
            save_data(data)
            st.success("Journal saved!")
            st.info(f"AI analysis: {sentiment}")

# --- My Journals ---
elif page == "📚 My Journals":
    st.subheader("📚 Your Past Journal Entries")
    entries = user["entries"]
    if entries:
        for e in reversed(entries):
            st.markdown(f"**Date:** {e['date']}")
            st.markdown(f"**Sentiment:** {e['sentiment']}")
            st.markdown(f"**Entry:** {e['text']}")
            st.markdown("---")
    else:
        st.info("You haven’t written any journals yet.")

# --- Graph ---
elif page == "📊 Graph":
    st.subheader("📊 Mood Over Time")
    entries = user["entries"]
    if len(entries) < 2:
        st.info("Not enough data for graph.")
    else:
        df = pd.DataFrame(entries)
        df['date'] = pd.to_datetime(df['date'])
        df['score'] = df['sentiment'].map({"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1})
        df = df.sort_values("date")
        st.line_chart(df.set_index("date")["score"])

# --- Insights ---
elif page == "🧠 Insights":
    st.header("🧠 Mood Overview")

    entries = user["entries"]
    if len(entries) < 2:
        st.info("Write more entries to view insights.")
    else:
        sentiments = [e["sentiment"] for e in entries]
        counts = pd.Series(sentiments).value_counts()
        st.bar_chart(counts)

        # Streak
        streak = 1
        for i in range(len(entries) - 2, -1, -1):
            d1 = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
            d2 = datetime.strptime(entries[i + 1]["date"], "%Y-%m-%d %H:%M").date()
            if (d2 - d1).days == 1:
                streak += 1
            else:
                break
        st.success(f"🔥 Current journaling streak: {streak} day(s)")

        # --- Print-style AI Summary ---
        st.subheader("🧠 AI Summary of Your Mood Trends (Generated Rules)")

        pos = sentiments.count("POSITIVE")
        neg = sentiments.count("NEGATIVE")
        neu = sentiments.count("NEUTRAL")

        if pos > neg and pos > neu:
            messages = [
                "You've mostly had an uplifting mood lately — keep riding that wave!",
                "Your positivity streak is going strong. Nice work on staying optimistic!",
                "It seems you're in a good mental space. Keep it up!"
            ]
        elif neg > pos and neg > neu:
            messages = [
                "You’ve been going through a tough time recently. Journaling helps — keep expressing yourself.",
                "Your emotional tone shows challenges — and strength. You're showing up even when it’s hard.",
                "Some heavy entries lately — remember, this space is here to support reflection."
            ]
        elif neu > pos and neu > neg:
            messages = [
                "You've been steady and balanced. Consistency can be powerful.",
                "Neutral emotions often mean calm — or deep thinking. Keep writing!",
                "Things have felt stable lately — nice job observing your mind."
            ]
        else:
            messages = [
                "Your moods are beautifully mixed — which is completely human.",
                "A healthy blend of thoughts and emotions is showing up. That’s a good sign.",
                "You're capturing a real, diverse emotional journey. That’s what journaling is for."
            ]

        st.markdown(f"**Summary:** {random.choice(messages)}")
