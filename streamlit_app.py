import streamlit as st
import json
import os
from datetime import datetime
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import pandas as pd

# Title and Tip
st.title("🧠 Memory Mirror - Smart AI Journal")
st.info("💡 Tip: If you enjoy writing your diary on paper but still want to use this app, you can use the Google Camera app (or Google Lens) to copy your handwritten text and paste it here.")

# === MODEL LOADERS (cached for performance) ===
@st.cache_resource
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

# === LOAD/CREATE JSON FILE ===
FILE = "entries.json"
if os.path.exists(FILE):
    with open(FILE, "r") as f:
        entries = json.load(f)
else:
    entries = []

# === TEXT INPUT ===
journal_text = st.text_area("Write your journal entry for today:", height=200)

# === PROCESSING ===
if st.button("Reflect"):
    if journal_text.strip() == "":
        st.warning("Please write something.")
    else:
        try:
            # Truncate very long text
            if len(journal_text.split()) > 500:
                journal_text = " ".join(journal_text.split()[:500])

            # Load AI tools
            sentiment_model = load_sentiment_model()
            summarizer = load_summarizer()
            embedder = load_embedder()

            # AI processing
            sentiment = sentiment_model(journal_text)[0]
            summary = summarizer(journal_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
            tags = [word.lower() for word in summary.split() if len(word) > 4]

            # Memory similarity check
            current_embedding = embedder.encode(journal_text, convert_to_tensor=True)
            reflection = None
            for entry in entries[::-1]:
                past_embedding = embedder.encode(entry["text"], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(current_embedding, past_embedding).item()
                if similarity > 0.7:
                    reflection = f"You had a similar entry on {entry['date']}."
                    break

            # Save new entry
            new_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "text": journal_text,
                "sentiment": sentiment['label'],
                "summary": summary,
                "tags": list(set(tags)),
            }
            entries.append(new_entry)
            with open(FILE, "w") as f:
                json.dump(entries, f, indent=2)

            # Show results
            st.success("✅ Entry saved and analyzed!")
            st.subheader("🧾 Summary")
            st.write(summary)

            st.subheader("😊 Mood")
            st.write(sentiment['label'])

            st.subheader("🏷️ Themes")
            st.write(", ".join(new_entry["tags"]))

            if reflection:
                st.subheader("🪞 Memory Mirror")
                st.info(reflection)

        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")

# === PAST ENTRIES ===
st.markdown("---")
st.header("📜 Past Entries")
for entry in entries[::-1]:
    with st.expander(entry["date"]):
        st.write(entry["text"])
        st.markdown(f"**Sentiment:** {entry['sentiment']}")
        st.markdown(f"**Summary:** {entry['summary']}")
        st.markdown(f"**Tags:** {', '.join(entry['tags'])}")

# === INSIGHTS FROM PAST ENTRIES ===
if entries:
    st.markdown("---")
    st.header("🧠 Insight Summary From Your Journal")

    # Mood trend
    sentiments = [e['sentiment'] for e in entries]
    pos = sentiments.count('POSITIVE')
    neg = sentiments.count('NEGATIVE')
    neu = len(sentiments) - pos - neg

    st.subheader("📊 Mood Overview")
    st.write(f"🟢 Positive: {pos} | 🔴 Negative: {neg} | ⚪ Neutral: {neu}")

    # Frequent tags
    tag_counter = Counter(tag for e in entries for tag in e['tags'])
    top_tags = tag_counter.most_common(5)
    if top_tags:
        st.subheader("🏷️ Most Frequent Themes")
        for tag, count in top_tags:
            st.write(f"- **{tag}** appeared in {count} entries")

    # Mood trend insight
    if len(sentiments) >= 3:
        recent = sentiments[-3:]
        if recent.count('NEGATIVE') >= 2:
            st.warning("😟 You've had multiple negative entries recently. Consider writing about what’s been bothering you.")
        elif recent.count('POSITIVE') >= 2:
            st.success("😊 You're trending positive lately — keep reflecting on what brings you joy!")

    # Mood line graph
    mood_scores = [1 if s == 'POSITIVE' else -1 if s == 'NEGATIVE' else 0 for s in sentiments]
    dates = [e['date'] for e in entries]
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Mood': mood_scores})
    df.set_index("Date", inplace=True)
    st.subheader("📈 Mood Over Time")
    st.line_chart(df)
                                                               
