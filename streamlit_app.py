import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import io
from fpdf import FPDF
from hashlib import sha256
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import random
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

def load_sentiment_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
        return pipe
    except Exception as e:
        st.error(f"❌ Failed to load sentiment model: {e}")
        return None

# Load sentiment model
sentiment_model = load_sentiment_model()
if sentiment_model:
    try:
        test = sentiment_model("I feel great today!")[0]
        st.sidebar.success(f"✅ AI model ready: {test['label']} ({test['score']:.2f})")
    except Exception as e:
        st.sidebar.error(f"Model test failed: {e}")
else:
    st.sidebar.error("⚠️ AI model not available.")

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
        "📝 New Entry", "📜 Past Journals", "🧠 Insights",
        "📊 Mood Graph", "📄 Download PDF", "💌 Future Note"
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
                elif sentiment_model:
                    try:
                        sentiment = sentiment_model(journal.strip())[0]
                        new_entry = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "text": journal,
                            "sentiment": sentiment["label"]
                        }
                        entries.append(new_entry)
                        save_entries(email, entries)
                        st.success("✅ Entry saved!")
                        st.markdown(f"**Sentiment:** {sentiment['label']}")
                    except Exception as e:
                        st.error(f"❌ AI analysis failed: {e}")
                else:
                    st.warning("AI model not available.")
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
st.subheader("🧠 AI Summary of Your Mood Trends")

if entries:
    sentiments = [e["sentiment"] for e in entries]
    total = len(sentiments)
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")
    neu = sentiments.count("NEUTRAL")

    insight_options = []

    if pos > neg and pos > neu:
        insight_options = [
            "You've been radiating mostly positive vibes — looks like things have been uplifting overall.",
            "Your recent entries suggest you're in a good headspace. Great to see that progress!",
            "There's a clear pattern of optimism. Keep doing whatever's working for you!",
            "You’ve captured a lot of meaningful highs recently. Keep journaling the good stuff!"
        ]
    elif neg > pos and neg > neu:
        insight_options = [
            "You've opened up about tough emotions lately. That's strength, not weakness.",
            "It seems you've had a rough stretch — remember, expressing it is part of healing.",
            "These reflections show you're moving through something hard. Keep writing through it.",
            "There’s been some emotional heaviness in your entries. It’s okay — you’re showing up for yourself."
        ]
    elif neu > pos and neu > neg:
        insight_options = [
            "Your emotional tone has been calm and steady. That says a lot about your self-awareness.",
            "You’ve been navigating things with balance — a reflective and thoughtful mood overall.",
            "A neutral rhythm shows you’re grounded. Continue checking in with yourself like this.",
            "These entries reveal a thoughtful equilibrium. Mindful journaling in action."
        ]
    else:
        insight_options = [
            "Your emotional landscape is varied — totally normal. This journal is capturing your real self.",
            "Some days are bright, others aren’t — and your entries reflect that beautifully.",
            "You're documenting both ups and downs — a powerful act of self-reflection.",
            "Mixed moods across entries show you're being real and honest with yourself. That's valuable."
        ]

    ai_summary = random.choice(insight_options)
    st.markdown(f"**AI Insight:** {ai_summary}")
else:
    st.info("Add a few more entries for an AI-generated summary.")
        st.header("🧠 Mood Overview")
        if len(entries) < 2:
            st.info("Write more entries to view insights.")
        else:
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

    # --- PDF Export ---
    elif page == "📄 Download PDF":
        st.header("📄 Export Your Journal")
        if not entries:
            st.info("No entries yet.")
        else:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for e in entries:
                pdf.add_page()
                pdf.multi_cell(0, 10, f"{e['date']}\n\nSentiment: {e['sentiment']}\n\n{e['text']}")
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button("Download PDF", data=pdf_output, file_name="my_journal.pdf", mime="application/pdf")

    # --- Note to Future ---
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
