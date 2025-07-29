import streamlit as st
import json
import os
from datetime import datetime, timedelta
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import pandas as pd
from fpdf import FPDF
import io

st.set_page_config(page_title="Memory Mirror", layout="wide")
st.title("Memory Mirror")

# Login
st.sidebar.header("Login")
username = st.sidebar.text_input("Your Name")
password = st.sidebar.text_input("Password", type="password")

if username and password:
    user_id = username.strip().lower().replace(" ", "_")
    filename = f"{user_id}_{password}.json"

    @st.cache_resource(show_spinner=False)
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @st.cache_resource(show_spinner=False)
    def load_summarizer():
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    @st.cache_resource(show_spinner=False)
    def load_embedder():
        return SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Load existing entries
    if os.path.exists(filename):
        with open(filename, "r") as f:
            entries = json.load(f)
    else:
        entries = []

    # Page selector
    page = st.sidebar.radio("Choose View", [
        "New Entry",
        "Past Journals",
        "Insights",
        "Mood Graph",
        "Download PDF",
        "Positive Entries"
    ])

    # New journal entry
    if page == "New Entry":
        st.header("Write your journal")

        # Daily prompt
        prompts = [
            "What made you smile today?",
            "What are you grateful for?",
            "What stressed you out today?",
            "What do you hope for tomorrow?"
        ]
        st.caption(f"Prompt: {prompts[datetime.now().day % len(prompts)]}")

        mood = st.radio("Your mood:", ["🙂", "😐", "😢", "😡", "😴"], horizontal=True)
        journal_text = st.text_area("Your entry", height=200)
        audio = st.file_uploader("Optional voice note (.wav)", type=["wav"])  # placeholder

        if st.button("Analyze"):
            if journal_text.strip():
                sentiment_model = load_sentiment_model()
                summarizer = load_summarizer()
                embedder = load_embedder()

                sentiment = sentiment_model(journal_text)[0]
                summary = summarizer(journal_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
                tags = [word.lower() for word in summary.split() if len(word) > 4]

                embedding = embedder.encode(journal_text, convert_to_tensor=True)
                reflection = None
                for entry in entries[::-1]:
                    past_embedding = embedder.encode(entry["text"], convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(embedding, past_embedding).item()
                    if similarity > 0.7:
                        reflection = f"This feels similar to your entry on {entry['date']}."
                        break

                new_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": journal_text,
                    "mood_emoji": mood,
                    "sentiment": sentiment['label'],
                    "summary": summary,
                    "tags": list(set(tags))
                }

                entries.append(new_entry)
                with open(filename, "w") as f:
                    json.dump(entries, f, indent=2)

                st.success("Entry saved.")
                st.subheader("Summary")
                st.write(summary)
                st.subheader("Sentiment")
                st.write(sentiment['label'])
                if reflection:
                    st.info(reflection)
            else:
                st.warning("Write something before analyzing.")

    # View past journals
    elif page == "Past Journals":
        st.header("Your Past Entries")
        if entries:
            for entry in reversed(entries):
                with st.expander(f"{entry['date']} {entry.get('mood_emoji','')}"):
                    st.write(entry["text"])
                    st.markdown(f"**Sentiment:** {entry['sentiment']}")
                    st.markdown(f"**Summary:** {entry['summary']}")
                    st.markdown(f"**Tags:** {', '.join(entry['tags'])}")
        else:
            st.info("No entries found.")

    # Insights summary
    elif page == "Insights":
        st.header("Insights from Previous Entries")
        if len(entries) < 2:
            st.info("Write more entries to see insights.")
        else:
            past_entries = entries[:-1]
            sentiments = [e["sentiment"] for e in past_entries]
            pos = sentiments.count("POSITIVE")
            neg = sentiments.count("NEGATIVE")
            neu = len(sentiments) - pos - neg

            st.write(f"Positive: {pos}, Negative: {neg}, Neutral: {neu}")

            recent = sentiments[-3:]
            if recent.count("NEGATIVE") >= 2:
                st.warning("You've had multiple negative entries recently.")
            elif recent.count("POSITIVE") >= 2 and recent.count("POSITIVE") > recent.count("NEGATIVE"):
                st.success("You're trending positive lately.")
            elif recent.count("POSITIVE") == recent.count("NEGATIVE"):
                st.info("Mixed mood trend.")

            # Streak counter
            streak = 1
            for i in range(len(entries)-2, -1, -1):
                date_i = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
                date_j = datetime.strptime(entries[i+1]["date"], "%Y-%m-%d %H:%M").date()
                if (date_j - date_i).days == 1:
                    streak += 1
                else:
                    break
            st.write(f"Current journaling streak: {streak} day(s)")

    # Mood graph
    elif page == "Mood Graph":
        st.header("Mood Over Time")
        if len(entries) >= 2:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood": [1 if e["sentiment"] == "POSITIVE" else -1 if e["sentiment"] == "NEGATIVE" else 0 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)
        else:
            st.info("Not enough entries for graph.")

    # Download PDF
    elif page == "Download PDF":
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for e in entries:
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"{e['date']}\n\nMood: {e.get('mood_emoji','')}\nSentiment: {e['sentiment']}\n\n{e['text']}\n\nSummary: {e['summary']}")
        buffer = io.BytesIO()
        pdf.output(buffer)
        st.download_button("Download Journal as PDF", buffer.getvalue(), "my_journal.pdf")

    # Positive entries only
    elif page == "Positive Entries":
        st.header("Positive Reflections")
        found = False
        for e in reversed(entries):
            if e["sentiment"] == "POSITIVE":
                st.success(f"{e['date']} - {e.get('mood_emoji','')}\n\n{e['summary']}")
                found = True
        if not found:
            st.info("No positive entries yet.")

else:
    st.warning("Please enter your name and password to start.")
                
