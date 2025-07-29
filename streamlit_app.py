import streamlit as st
from transformers import pipeline
import datetime
import json
import os

st.set_page_config(page_title="Memory Mirror", layout="centered")

# Load sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

# File to store journal entries
FILE = "entries.json"

# Load previous entries
def load_entries():
    if os.path.exists(FILE):
        with open(FILE, 'r') as f:
            return json.load(f)
    return {}

# Save a new entry
def save_entry(date, text, mood, keywords):
    data = load_entries()
    data[date] = {
        'text': text,
        'mood': mood,
        'keywords': keywords
    }
    with open(FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Analyze mood
def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return f"{result['label']} ({round(result['score'], 2)})"

# Extract simple keywords
def extract_keywords(text):
    words = text.lower().split()
    stopwords = set(['i', 'am', 'the', 'and', 'a', 'is', 'in', 'to', 'of', 'my', 'it', 'for', 'on', 'with'])
    keywords = [w.strip(".,?!") for w in words if w not in stopwords and len(w) > 3]
    return list(set(keywords))[:5]

# Streamlit UI
st.title("🪞 Memory Mirror")
st.markdown("Write. Reflect. Understand yourself.")

menu = ["Write Entry", "View Reflections"]
choice = st.sidebar.selectbox("Select an option", menu)

if choice == "Write Entry":
    st.subheader("📘 Daily Journal Entry")
    text = st.text_area("What’s on your mind today?", height=200)

    if st.button("Analyze & Save"):
        if text.strip():
            mood = analyze_sentiment(text)
            keywords = extract_keywords(text)
            today = str(datetime.date.today())
            save_entry(today, text, mood, keywords)

            st.success(f"Entry saved for {today}")
            st.write(f"🧠 **Mood**: {mood}")
            st.write(f"🔑 **Keywords**: {', '.join(keywords)}")
        else:
            st.warning("Please enter some text.")

elif choice == "View Reflections":
    st.subheader("📖 Past Reflections")
    data = load_entries()

    if not data:
        st.info("No entries yet.")
    else:
        all_keywords = {}
        for date, entry in sorted(data.items(), reverse=True):
            st.markdown(f"### {date}")
            st.markdown(f"**Mood**: {entry['mood']}")
            st.markdown(f"**Keywords**: {', '.join(entry['keywords'])}")
            st.markdown(f"_Entry_: {entry['text']}")
            st.markdown("---")

            for kw in entry['keywords']:
                all_keywords[kw] = all_keywords.get(kw, 0) + 1

        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_keywords:
            st.markdown("### 💡 Top 5 Recurring Thoughts:")
            for word, freq in top_keywords:
                st.write(f"- {word} ({freq} times)")
