import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import torch
from transformers import pipeline
from fpdf import FPDF
import io

st.set_page_config(page_title="Memory Mirror", layout="wide")
st.title("🧠 Memory Mirror - AI Journal")

# === LOGIN / SIGNUP ===
st.sidebar.header("🔐 Login / Signup")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

auth_mode = st.sidebar.radio("Mode", ["Login", "Sign Up"])
input_user = st.sidebar.text_input("Username")
input_pass = st.sidebar.text_input("Password", type="password")
auth_btn = st.sidebar.button("Continue")

if auth_btn:
    if input_user and input_pass:
        user_id = input_user.strip().lower().replace(" ", "_")
        filename = f"{user_id}_{input_pass}.json"
        if auth_mode == "Login":
            if os.path.exists(filename):
                st.session_state.logged_in = True
                st.session_state.username = input_user
                st.session_state.password = input_pass
                st.success("✅ Logged in!")
                st.rerun()
            else:
                st.sidebar.error("❌ No account found.")
        else:  # Sign Up
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    json.dump([], f)
                st.success("✅ Account created.")
            else:
                st.warning("Account already exists.")
    else:
        st.sidebar.warning("Please fill in both fields.")

# === MAIN APP ===
if st.session_state.logged_in:
    username = st.session_state.username
    password = st.session_state.password
    user_id = username.strip().lower().replace(" ", "_")
    filename = f"{user_id}_{password}.json"

    # Load entries
    if os.path.exists(filename):
        with open(filename, "r") as f:
            entries = json.load(f)
    else:
        entries = []

    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis")

    sentiment_model = load_sentiment_model()

    # Navigation
    page = st.sidebar.radio("Navigate", [
        "📝 New Entry",
        "📜 Past Journals",
        "🧠 Insights",
        "📊 Mood Graph",
        "📄 Download PDF",
        "🌈 Uplifting Feed"
    ])

    # === New Entry ===
    if page == "📝 New Entry":
        st.header("Write your journal entry")
        journal_text = st.text_area("What's on your mind today?", height=200)
        if st.button("Save & Analyze"):
            if journal_text.strip():
                sentiment = sentiment_model(journal_text)[0]
                new_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": journal_text,
                    "sentiment": sentiment["label"]
                }
                entries.append(new_entry)
                with open(filename, "w") as f:
                    json.dump(entries, f, indent=2)
                st.success("✅ Entry saved!")
                st.markdown(f"**Sentiment detected:** {sentiment['label']}")
            else:
                st.warning("Please write something.")

    # === Past Journals ===
    elif page == "📜 Past Journals":
        st.header("📜 Your Entries")
        for e in reversed(entries):
            with st.expander(e["date"]):
                st.write(e["text"])
                st.markdown(f"**Sentiment:** {e['sentiment']}")

    # === Insights ===
    elif page == "🧠 Insights":
        st.header("📊 Mood Trends")
        if len(entries) < 2:
            st.info("Write more entries to see insights.")
        else:
            sentiments = [e["sentiment"] for e in entries[:-1]]
            pos = sentiments.count("POSITIVE")
            neg = sentiments.count("NEGATIVE")
            neu = len(sentiments) - pos - neg
            st.write(f"🟢 Positive: {pos} | 🔴 Negative: {neg} | ⚪ Neutral: {neu}")

            # Streak tracker
            streak = 1
            for i in range(len(entries)-2, -1, -1):
                d1 = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
                d2 = datetime.strptime(entries[i+1]["date"], "%Y-%m-%d %H:%M").date()
                if (d2 - d1).days == 1:
                    streak += 1
                else:
                    break
            st.info(f"🔥 Current journaling streak: {streak} day(s)")

    # === Mood Graph ===
    elif page == "📊 Mood Graph":
        st.header("📈 Mood Over Time")
        if len(entries) < 2:
            st.info("Not enough data yet.")
        else:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood": [1 if e["sentiment"] == "POSITIVE" else -1 if e["sentiment"] == "NEGATIVE" else 0 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)

    # === Download PDF ===
    elif page == "📄 Download PDF":
        st.header("📥 Download Your Journal")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for e in entries:
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"{e['date']}\n\nSentiment: {e['sentiment']}\n\n{e['text']}")
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        st.download_button("Download as PDF", pdf_output.getvalue(), "my_journal.pdf")

    # === Uplifting Feed ===
    elif page == "🌈 Uplifting Feed":
        st.header("🌈 Uplifting Moments")
        positives = [e for e in entries if e["sentiment"] == "POSITIVE"]
        if not positives:
            st.info("No positive entries yet. Keep writing!")
        else:
            for e in reversed(positives):
                st.success(f"{e['date']}  \n{e['text'][:200]}...")
else:
    st.info("🔐 Please log in to access the journal.")
