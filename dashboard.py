"""Analysis dashboard for a recorded session.  Run:  streamlit run dashboard.py"""
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ActiSense Analysis", layout="wide")
st.title("ActiSense – Activity Analysis")

path = st.sidebar.text_input("Session log (CSV)", "logs/session.csv")
try:
    df = pd.read_csv(path, parse_dates=["timestamp"])
except FileNotFoundError:
    st.error(f"No log found at {path}. Run run_app.py first.")
    st.stop()

df = df[df["activity"] != "unknown"]
if df.empty:
    st.warning("The log has no recognised activity yet.")
    st.stop()

falls = int(((df["fall_alert"].diff() == 1) | ((df.index == df.index[0]) & (df["fall_alert"] == 1))).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active duration", f"{len(df)} s")
c2.metric("Activities seen", df["activity"].nunique())
c3.metric("Squat reps", int(df["squat_reps"].max()))
c4.metric("Fall alerts", falls)

st.subheader("Time spent per activity")
per = df["activity"].value_counts().rename_axis("activity").reset_index(name="seconds")
st.altair_chart(alt.Chart(per).mark_bar().encode(
    x="seconds:Q", y=alt.Y("activity:N", sort="-x"), color="activity:N"), use_container_width=True)

st.subheader("Activity timeline")
st.altair_chart(alt.Chart(df).mark_tick(thickness=4, size=25).encode(
    x="timestamp:T", y="activity:N", color="activity:N",
    tooltip=["timestamp:T", "activity:N", "confidence:Q"]), use_container_width=True)

st.subheader("Activity transitions")
seg = df[df["activity"] != df["activity"].shift()]
trans = pd.DataFrame({"from": seg["activity"].shift().iloc[1:], "to": seg["activity"].iloc[1:]})
if trans.empty:
    st.write("No transitions in this session.")
else:
    st.dataframe(trans.value_counts().reset_index(name="count"), use_container_width=True)

st.subheader("Average model confidence per activity")
st.dataframe(df.groupby("activity")["confidence"].mean().round(2).reset_index(), use_container_width=True)

with st.expander("Raw log"):
    st.dataframe(df, use_container_width=True)
