import streamlit as st
import json
import os
from datetime import datetime
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

st.title("🧠 Memory Mirror - Smart AI Journal")

st.info("💡 Tip: If you enjoy writing your diary on paper but still want to use this app, you can use the Google Camera app (or Google Lens) to copy your handwritten text and paste it here.")

# Lazy model loading functions
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
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=device
    )

@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load or create entry file
FILE = "entries.json"
if os.path.exists(FILE):
    with open(FILE, "r") as f:
        entries = json.load(f)
else:
    entries = []

# Input
journal_text = st.text_area("Write your journal entry for today:", height=200)

if st.button("Reflect"):
    if journal_text.strip() == "":
        st.warning("Please write something.")
    else:
        try:
            # Truncate long entries
            if len(journal_text.split()) > 500:
                journal_text = " ".join(journal_text.split()[:500])

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
                "sentiment": sentiment['label'],
                "summary": summary,
                "tags": list(set(tags)),
            }
            entries.append(new_entry)
            with open(FILE, "w") as f:
                json.dump(entries, f, indent=2)

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
            st.error(f"⚠️ Error while analyzing your journal: {e}")

# Past entries section
st.markdown("---")
st.header("📜 Past Entries")
for entry in entries[::-1]:
    with st.expander(entry["date"]):
        st.write(entry["text"])
        st.markdown(f"**Sentiment:** {entry['sentiment']}")
        st.markdown(f"**Summary:** {entry['summary']}")
        st.markdown(f"**Tags:** {', '.join(entry['tags'])}")
        
