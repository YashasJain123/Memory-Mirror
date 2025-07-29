import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import io
from fpdf import FPDF
from hashlib import sha256
from transformers import pipeline

# --- Setup ---
st.set_page_config("Memory Mirror", layout="wide")
st.title("🧠 Memory Mirror - AI-Powered Journal")

# --- File paths ---
USERS_FILE = "users.json"

# --- Utilities ---
def get_email_hash(email):
    return sha256(email.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_entries(email):
    file = f"{get_email_hash(email)}.json"
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return []

def save_entries(email, entries):
    file = f"{get_email_hash(email)}.json"
    with open(file, "w") as f:
        json.dump(entries, f, indent=2)

def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

# --- Login System ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.header("🔐 Login / Sign Up")
    mode = st.sidebar.radio("Mode", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    users = load_users()

    if st.sidebar.button("Continue"):
        if not email or not password:
            st.sidebar.error("Enter both email and password.")
        elif mode == "Login":
            if email in users and users[email] == password:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.rerun()
            else:
                st.sidebar.error("Incorrect credentials.")
        else:  # Sign Up
            if email in users:
                st.sidebar.warning("Account already exists.")
            else:
                users[email] = password
                save_users(users)
                st.success("Account created. Please log in.")
                st.rerun()

# --- Main App ---
if st.session_state.get("logged_in"):
    email = st.session_state.email
    entries = load_entries(email)

    if "name" not in st.session_state:
        st.session_state.name = st.text_input("What should we call you?", placeholder="e.g. Aanya")
        if st.session_state.name:
            st.success(f"Hi {st.session_state.name}, welcome to your journal!")
            st.rerun()
        else:
            st.stop()

    name = st.session_state.name
    page = st.sidebar.radio("Navigate", [
        "📝 New Entry", "📜 Past Journals", "🧠 Insights", "📊 Mood Graph", "📄 Download PDF", "💌 Future Note"
    ])

    # === NEW ENTRY ===
    if page == "📝 New Entry":
        st.header(f"Dear {name}, what’s on your mind today?")
        st.markdown("💡 *Tip: If you like to write your diary on paper and still want to use this app, use Google Camera (or any scanner) to copy the text and paste it here.*")
        journal = st.text_area("Start writing here...", height=200)

        if st.button("Save & Analyze"):
            if journal.strip():
                sentiment = sentiment_model(journal)[0]
                new_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": journal,
                    "sentiment": sentiment["label"]
                }
                entries.append(new_entry)
                save_entries(email, entries)
                st.success("✅ Entry saved!")
                st.markdown(f"**Sentiment:** {sentiment['label']}")
            else:
                st.warning("Please write something.")

    # === PAST JOURNALS ===
    elif page == "📜 Past Journals":
        st.header("📜 Your Journal Entries")
        for e in reversed(entries):
            with st.expander(e["date"]):
                st.write(e["text"])
                st.markdown(f"**Sentiment:** {e['sentiment']}")

    # === INSIGHTS ===
    elif page == "🧠 Insights":
        st.header("🧠 Mood Overview")
        if len(entries) < 2:
            st.info("Write more entries to view insights.")
        else:
            sentiments = [e["sentiment"] for e in entries]
            counts = pd.Series(sentiments).value_counts()
            st.write("Mood summary based on all entries:")
            st.bar_chart(counts)

            # Streak
            streak = 1
            for i in range(len(entries)-2, -1, -1):
                d1 = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
                d2 = datetime.strptime(entries[i+1]["date"], "%Y-%m-%d %H:%M").date()
                if (d2 - d1).days == 1:
                    streak += 1
                else:
                    break
            st.info(f"🔥 Current journaling streak: {streak} day(s)")

    # === MOOD GRAPH ===
    elif page == "📊 Mood Graph":
        st.header("📊 Mood Over Time")
        if len(entries) < 2:
            st.info("Not enough entries for a graph.")
        else:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood Score": [1 if e["sentiment"] == "POSITIVE" else -1 if e["sentiment"] == "NEGATIVE" else 0 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)

    # === DOWNLOAD PDF ===
    elif page == "📄 Download PDF":
        st.header("📄 Export Your Journal as PDF")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for e in entries:
            pdf.add_page()
            pdf.multi_cell(0, 10, f"{e['date']}\n\nSentiment: {e['sentiment']}\n\n{e['text']}")
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button("Download PDF", data=pdf_buffer, file_name="my_journal.pdf", mime="application/pdf")

    # === FUTURE NOTE ===
    elif page == "💌 Future Note":
        st.header("💌 Write a Note to Future You")
        future_file = f"{get_email_hash(email)}_future.json"
        if os.path.exists(future_file):
            with open(future_file, "r") as f:
                note = json.load(f)
                reveal_date = datetime.strptime(note["reveal_date"], "%Y-%m-%d")
                if datetime.now().date() >= reveal_date.date():
                    st.success("🗓️ Here's your note to yourself:")
                    st.markdown(f"**Note written on {note['written_on']}:**  \n{note['text']}")
                else:
                    st.info(f"⏳ Your note will unlock on **{reveal_date.date()}**.")
        else:
            note_text = st.text_area("What would you like to tell your future self?")
            days = st.slider("Reveal after (days)", 1, 30, 7)
            if st.button("Save Note"):
                note = {
                    "text": note_text,
                    "written_on": datetime.now().strftime("%Y-%m-%d"),
                    "reveal_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                }
                with open(future_file, "w") as f:
                    json.dump(note, f)
                st.success(f"✅ Saved! Your note will unlock in {days} day(s).")
