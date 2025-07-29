
import streamlit as st
import json
import os
from datetime import datetime, timedelta
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Memory Mirror", layout="wide")
st.title("🧠 Memory Mirror - Secure AI Journal")

# Sidebar login
st.sidebar.header("🔐 Login")
username = st.sidebar.text_input("Your Name")
password = st.sidebar.text_input("Your Password", type="password")

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

    if os.path.exists(filename):
        with open(filename, "r") as f:
            entries = json.load(f)
    else:
        entries = []

    page = st.sidebar.radio("📂 Select View", [
        "📝 New Entry",
        "📜 View Past Journals",
        "🧠 Insights Summary",
        "📊 Mood Graph",
        "📄 Download PDF",
        "🧱 Positive Wall"
    ])

    if page == "📝 New Entry":
        st.header(f"Hi {username.title()}, write your journal entry:")
        prompts = [
            "What made you smile today?",
            "Is there something you’re grateful for?",
            "What drained your energy today?",
            "Describe one moment that stood out today."
        ]
        st.markdown(f"💬 Prompt: *{prompts[datetime.now().day % len(prompts)]}*")
        mood = st.radio("How are you feeling today?", ["😊", "😐", "😢", "😠", "😴"], horizontal=True)
        journal_text = st.text_area("Your Journal", height=200)
        audio = st.file_uploader("Optional: Upload voice note (.wav)", type=["wav"])

        if st.button("Reflect & Analyze"):
            if journal_text.strip() == "":
                st.warning("Please write something.")
            else:
                sentiment_model = load_sentiment_model()
                summarizer = load_summarizer()
                embedder = load_embedder()

                sentiment = sentiment_model(journal_text)[0]
                summary = summarizer(journal_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
                tags = [word.lower() for word in summary.split() if len(word) > 4]
                current_embedding = embedder.encode(journal_text, convert_to_tensor=True)
                reflection = None
                for entry in entries[::-1]:
                    past_embedding = embedder.encode(entry["text"], convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(current_embedding, past_embedding).item()
                    if similarity > 0.7:
                        reflection = f"You had a similar entry on {entry['date']}."
                        break

                new_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": journal_text,
                    "mood_emoji": mood,
                    "sentiment": sentiment['label'],
                    "summary": summary,
                    "tags": list(set(tags)),
                }
                entries.append(new_entry)
                with open(filename, "w") as f:
                    json.dump(entries, f, indent=2)

                st.success("✅ Entry saved!")
                st.subheader("🧾 Summary")
                st.write(summary)
                st.subheader("😊 Sentiment")
                st.write(sentiment['label'])
                st.subheader("🏷️ Themes")
                st.write(", ".join(new_entry["tags"]))
                if reflection:
                    st.subheader("🪞 Memory Mirror")
                    st.info(reflection)

    elif page == "📜 View Past Journals":
        st.header(f"📜 {username.title()}'s Journal")
        for entry in entries[::-1]:
            with st.expander(f"{entry['date']} {entry.get('mood_emoji', '')}"):
                st.write(entry["text"])
                st.markdown(f"**Sentiment:** {entry['sentiment']}")
                st.markdown(f"**Summary:** {entry['summary']}")
                st.markdown(f"**Tags:** {', '.join(entry['tags'])}")

    elif page == "🧠 Insights Summary":
        st.header("🧠 Insights (Excludes today's entry)")
        if len(entries) < 2:
            st.info("Add more entries to view insights.")
        else:
            past_entries = entries[:-1]
            sentiments = [e["sentiment"] for e in past_entries]
            pos = sentiments.count("POSITIVE")
            neg = sentiments.count("NEGATIVE")
            neu = len(sentiments) - pos - neg
            st.subheader("📊 Mood Distribution")
            st.write(f"🟢 Positive: {pos} | 🔴 Negative: {neg} | ⚪ Neutral: {neu}")

            recent = sentiments[-3:]
            if recent.count("NEGATIVE") >= 2:
                st.warning("😟 You've had multiple negative entries recently.")
            elif recent.count("POSITIVE") >= 2 and recent.count("POSITIVE") > recent.count("NEGATIVE"):
                st.success("😊 You're trending positively lately!")
            elif recent.count("POSITIVE") == recent.count("NEGATIVE"):
                st.info("😐 Mixed mood trend.")

            streak = 1
            for i in range(len(entries)-2, -1, -1):
                date_i = datetime.strptime(entries[i]["date"], "%Y-%m-%d %H:%M").date()
                date_j = datetime.strptime(entries[i+1]["date"], "%Y-%m-%d %H:%M").date()
                if (date_j - date_i).days == 1:
                    streak += 1
                else:
                    break
            st.subheader(f"🔥 Current journaling streak: {streak} day(s)")

    elif page == "📊 Mood Graph":
        st.header("📈 Mood Over Time")
        if len(entries) < 2:
            st.info("Not enough entries for a graph.")
        else:
            df = pd.DataFrame({
                "Date": [pd.to_datetime(e["date"]) for e in entries],
                "Mood": [1 if e["sentiment"]=="POSITIVE" else -1 if e["sentiment"]=="NEGATIVE" else 0 for e in entries]
            }).sort_values("Date")
            df.set_index("Date", inplace=True)
            st.line_chart(df)

    elif page == "📄 Download PDF":
        import io
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for e in entries:
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"{e['date']}

Mood: {e.get('mood_emoji','')}
Sentiment: {e['sentiment']}

{e['text']}

Summary: {e['summary']}")
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        st.download_button("📥 Download All Entries as PDF", pdf_output.getvalue(), "my_journal.pdf")

    elif page == "🧱 Positive Wall":
        st.header("🌈 Your Positive Memory Wall")
        for e in reversed(entries):
            if e["sentiment"] == "POSITIVE":
                st.success(f"🗓 {e['date']} {e.get('mood_emoji','')}  
{e['summary']}")

else:
    st.warning("🔐 Please log in with name & password to access your journal.")
        
