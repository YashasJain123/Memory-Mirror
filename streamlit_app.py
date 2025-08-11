
import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from fpdf import FPDF
from hashlib import sha256
from transformers import pipeline

# --- Setup ---
st.set_page_config("Memory Mirror", layout="wide")
st.title("🧠 Memory Mirror - AI-Powered Journal")

USERS_FILE = "users.json"

def get_email_hash(email):
    return sha256(email.encode()).hexdigest()

def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(users):
    json.dump(users, open(USERS_FILE, "w"))

def load_entries(email):
    file = f"{get_email_hash(email)}.json"
    return json.load(open(file)) if os.path.exists(file) else []

def save_entries(email, entries):
    file = f"{get_email_hash(email)}.json"
    json.dump(entries, open(file, "w"), indent=2)

def analyze_sentiment(text):
    try:
        sentiment_model = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1  # Force CPU for Streamlit Cloud
        )
        result = sentiment_model(text, candidate_labels=["positive", "negative", "neutral"])
        return result["labels"][0].upper()
    except Exception as e:
        st.error(f"❌ AI analysis failed: {e}")
        return "UNKNOWN"

# --- Auth ---
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
        else:
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
        "📝 New Entry", "📜 Past Journals", "🧠 Insights", "📊 Mood Graph", "💌 Future Note"
    ])

    # --- Journal Entry ---
    if page == "📝 New Entry":
        st.header(f"Dear {name}, what’s on your mind today?")
        st.markdown("💡 *Tip: If you like to write your diary on paper and still want to use this app, use Google Camera (or any scanner) to copy the text and paste it here.*")
        journal = st.text_area("Start writing here...", height=200)

        if st.button("Save & Analyze"):
            if journal.strip():
                if len(journal.split()) < 10:
                    st.warning("✍️ Journal is too short. Try writing at least 10 words.")
                else:
                    sentiment = analyze_sentiment(journal.strip())
                    new_entry = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "text": journal,
                        "sentiment": sentiment
                    }
                    entries.append(new_entry)
                    save_entries(email, entries)
                    st.success("✅ Entry saved!")
                    st.markdown(f"**Sentiment:** {sentiment}")
            else:
                st.warning("Please write something.")

    # --- Past Journals ---
    elif page == "📜 Past Journals":
        st.header("📜 Your Journal Entries")
        for e in reversed(entries):
            with st.expander(e["date"]):
                st.write(e["text"])
                st.markdown(f"**Sentiment:** {e['sentiment']}")

    # --- Insights ---
    elif page == "🧠 Insights":
        st.header("🧠 Mood Overview")
        if len(entries) < 2:
            st.info("Write more entries to view insights.")
        else:
            import pandas as pd
            sentiments = [e["sentiment"] for e in entries]
            counts = pd.Series(sentiments).value_counts()
            st.bar_chart(counts)

            streak = 1
            for i in range(len(entries) - 2, -1, -1):
                d1 = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
                d2 = datetime.strptime(entries[i+1]["date"], "%Y-%m-%d %H:%M").date()
                if (d2 - d1).days == 1:
                    streak += 1
                else:
                    break
            st.success(f"🔥 Current journaling streak: {streak} day(s)")

    # --- Mood Graph ---
    elif page == "📊 Mood Graph":
        st.header("📊 Mood Over Time")
        if len(entries) < 2:
            st.info("Not enough entries for graph.")
        else:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood Score": [1 if e["sentiment"] == "POSITIVE" else -1 if e["sentiment"] == "NEGATIVE" else 0 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)

    # --- Future Note ---
    elif page == "💌 Future Note":
        st.header("💌 Message to Future You")
        future_file = f"{get_email_hash(email)}_future.json"

        if os.path.exists(future_file):
            with open(future_file, "r") as f:
                note = json.load(f)
                reveal_date = datetime.strptime(note["reveal_date"], "%Y-%m-%d")
                if datetime.now().date() >= reveal_date.date():
                    st.success(f"🗓️ Note from {note['written_on']} unlocked:")
                    st.markdown(note["text"])
                else:
                    st.info(f"⏳ This note will unlock on **{note['reveal_date']}**.")
        else:
            choice = st.radio("How do you want to create the note?", ["Write my own", "Generate by AI"])
            days = st.slider("Reveal after (days)", 1, 30, 7)

            if choice == "Write my own":
                note_text = st.text_area("Write your message here...")
            else:
                sentiments = [e["sentiment"] for e in entries]
                pos = sentiments.count("POSITIVE")
                neg = sentiments.count("NEGATIVE")
                note_text = f"Hey {name}, you've written {len(entries)} entries. You've had {pos} positive and {neg} tough days. You're doing great — keep going 💪"

            if st.button("Save Note"):
                note = {
                    "text": note_text,
                    "written_on": datetime.now().strftime("%Y-%m-%d"),
                    "reveal_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                }
                with open(future_file, "w") as f:
                    json.dump(note, f)
                st.success("✅ Your note is saved and will unlock on the selected day.")
