import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from hashlib import sha256
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    pipeline
)

# --- Page Config ---
st.set_page_config("🧠 Memory Mirror - AI Journal", layout="wide")
st.title("🧠 Memory Mirror - AI Journal")

# --- File Paths ---
USERS_FILE = "users.json"

# --- Utility Functions ---
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

# --- AI Model Loaders ---
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
        return TextClassificationPipeline(model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"❌ Failed to load sentiment model: {e}")
        return None

@st.cache_resource
def load_reflection_model():
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception as e:
        st.error(f"❌ Failed to load GPT-2 model: {e}")
        return None

sentiment_model = load_sentiment_model()
reflection_model = load_reflection_model()

if sentiment_model:
    try:
        test = sentiment_model("I feel great today!")[0]
        st.sidebar.success(f"✅ AI model ready: {test['label']} ({test['score']:.2f})")
    except:
        st.sidebar.error("❌ Sentiment model test failed.")
else:
    st.sidebar.error("⚠️ AI sentiment model not available.")

# --- Auth System ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.header("🔐 Login / Sign Up")
    mode = st.sidebar.radio("Choose", ["Login", "Sign Up"])
    email = st.sidebar.text_input("📧 Email")
    password = st.sidebar.text_input("🔑 Password", type="password")
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
                st.success("Account created! Please log in.")
                st.rerun()

# --- Main App ---
if st.session_state.logged_in:
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

    page = st.sidebar.radio("📁 Menu", [
        "📝 New Entry", "📜 Past Journals", "📊 Mood Graph",
        "🧠 Insights", "💌 Future Note", "💬 Deep Journal Insight (AI)"
    ])

    # --- New Entry ---
    if page == "📝 New Entry":
        st.header(f"📝 Dear {name}, what’s on your mind today?")
        st.markdown("💡 *Tip: If you write on paper, you can scan with Google Camera and paste here.*")

        journal = st.text_area("Start writing here...", height=200)

        if st.button("Save & Analyze"):
            if not journal.strip():
                st.warning("✍️ Please write something.")
            elif len(journal.split()) < 10:
                st.warning("Your journal is too short. Try at least 10 words.")
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

    # --- Past Journals ---
    elif page == "📜 Past Journals":
        st.header("📜 Your Journal Entries")
        if not entries:
            st.info("No entries yet.")
        for e in reversed(entries):
            with st.expander(e["date"]):
                st.write(e["text"])
                st.markdown(f"**Sentiment:** {e['sentiment']}")

    # --- Mood Graph ---
    elif page == "📊 Mood Graph":
        st.header("📊 Mood Over Time")
        if len(entries) < 2:
            st.info("Not enough data for a graph.")
        else:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood Score": [1 if e["sentiment"] == "POSITIVE" else -1 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)

    # --- Insights ---
    elif page == "🧠 Insights":
        st.header("🧠 Mood Insights")
        if len(entries) < 2:
            st.info("Write a few more entries to generate insights.")
        else:
            sentiments = [e["sentiment"] for e in entries]
            counts = pd.Series(sentiments).value_counts()
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
            st.success(f"🔥 Current journaling streak: {streak} day(s)")

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
            choice = st.radio("Create note by", ["✍️ Write my own", "🤖 Generate with AI"])
            days = st.slider("Reveal after (days)", 1, 30, 7)

            if choice == "✍️ Write my own":
                note_text = st.text_area("Write your message here...")
            else:
                pos = sum(1 for e in entries if e["sentiment"] == "POSITIVE")
                neg = sum(1 for e in entries if e["sentiment"] == "NEGATIVE")
                note_text = f"Hey {name}, you've written {len(entries)} entries. {pos} were positive, {neg} were difficult. Keep growing 💪"

            if st.button("Save Note"):
                note = {
                    "text": note_text,
                    "written_on": datetime.now().strftime("%Y-%m-%d"),
                    "reveal_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                }
                with open(future_file, "w") as f:
                    json.dump(note, f)
                st.success("✅ Your note is saved!")

    # --- Deep Journal Insight ---
    elif page == "💬 Deep Journal Insight (AI)":
        st.header("💬 Deep Journal Insight")

        if not entries:
            st.warning("You need at least one journal entry.")
            st.stop()

        all_text = " ".join([e["text"] for e in entries if e["text"]])
        max_chunk_size = 512
        chunks = [all_text[i:i+max_chunk_size] for i in range(0, len(all_text), max_chunk_size)]

        with st.spinner("Analyzing your emotions..."):
            try:
                results = [sentiment_model(chunk)[0] for chunk in chunks]
                pos = sum(1 for r in results if r["label"] == "POSITIVE")
                neg = sum(1 for r in results if r["label"] == "NEGATIVE")
                neu = len(results) - pos - neg

                st.subheader("📊 Sentiment Breakdown")
                st.markdown(f"✅ Positive: **{pos}**")
                st.markdown(f"❌ Negative: **{neg}**")
                st.markdown(f"➖ Neutral/Other: **{neu}**")
            except Exception as e:
                st.error(f"❌ AI analysis failed: {e}")

        # --- GPT-2 AI Reflection ---
        st.subheader("🧠 AI Reflection (Generated)")

if reflection_model:
    recent_text = " ".join([e["text"] for e in entries[-3:]])
    prompt = f"Reflect on this person's emotional journey:\n{recent_text}\nReflection:"
    
    try:
        with st.spinner("🤖 Generating personalized insight..."):
            reflection = reflection_model(prompt, max_length=100)[0]['generated_text']
            reflection = reflection.split("Reflection:")[-1].strip()
            st.success(reflection)
    
    except Exception as e:
        st.error(f"❌ AI reflection failed: {e}")

else:
    st.info("⚠️ GPT-2 model not available.")
