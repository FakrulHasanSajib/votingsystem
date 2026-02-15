import streamlit as st
import pandas as pd
import plotly.express as px
import os

# পেজ কনফিগারেশন
st.set_page_config(page_title="Live Election Dashboard", layout="wide")

# অটো-রিফ্রেশ সেটআপ (প্রতি ২ সেকেন্ড পর পর আপডেট হবে)
st.logo("https://cdn-icons-png.flaticon.com/512/927/927295.png")  # ঐচ্ছিক: একটি লোগো
st.title("🗳️ Real-Time Election Results Dashboard")
st.markdown("---")


# ডেটা পড়ার ফাংশন
def get_results():
    votes = {"Candidate A": 0, "Candidate B": 0, "Candidate C": 0}
    if os.path.exists("election_results.txt"):
        try:
            with open("election_results.txt", "r") as f:
                for line in f:
                    if ":" in line:
                        name, count = line.strip().split(":")
                        votes[name] = int(count)
        except Exception as e:
            st.error(f"Error reading data: {e}")
    return votes


# ডেটা লোড করা
data = get_results()
df = pd.DataFrame(list(data.items()), columns=['Candidate', 'Votes'])

# ১. টপ কার্ডস (KPIs)
total_votes = df['Votes'].sum()
winner_row = df.loc[df['Votes'].idxmax()]

col1, col2, col3 = st.columns(3)
col1.metric("Total Votes Cast", total_votes)
col2.metric("Current Leader", winner_row['Candidate'], f"{winner_row['Votes']} votes")
col3.metric("Active Booths", "1 (Local)")

st.markdown("---")

# ২. গ্রাফ এবং টেবিল সেকশন
left_column, right_column = st.columns([2, 1])

with left_column:
    st.markdown("### 📊 Vote Distribution")
    fig = px.bar(df, x='Candidate', y='Votes', color='Candidate',
                 text='Votes', color_discrete_sequence=px.colors.qualitative.Set2)

    # এখানে 'key' অ্যাড করা হয়েছে এরর ফিক্স করার জন্য
    st.plotly_chart(fig, use_container_width=True, key="election_chart_unique")

with right_column:
    st.markdown("### 📋 Detailed Tally")
    st.dataframe(df, hide_index=True, use_container_width=True)

# ৩. অটো রিফ্রেশ লজিক (Streamlit এর আধুনিক নিয়ম)
st.info("The dashboard updates automatically every 2 seconds.")
time_interval = 2
st.empty()  # স্ক্রিন ক্লিনার
import time

time.sleep(time_interval)
st.rerun()  # এটি পুরো পেজকে রিফ্রেশ করবে