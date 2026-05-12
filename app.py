import streamlit as st
import requests

st.set_page_config(page_title="SHL Consultant", layout="centered")

st.title("💼 SHL Solutions Consultant")
st.markdown("Enter your hiring needs below to get a tailored assessment stack.")

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# RENDER BACKEND URL (IMPORTANT FIX)
# ----------------------------
API_URL = "https://shl-consultant-bot.onrender.com/chat"

# ----------------------------
# USER INPUT
# ----------------------------
if prompt := st.chat_input("How can I help with your talent audit?"):

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Payload for FastAPI
    payload = {
        "messages": st.session_state.messages
    }

    try:
        # Call Render backend (NOT localhost anymore)
        response = requests.post(API_URL, json=payload, timeout=60)

        data = response.json()

        ai_reply = data.get("reply", "No reply received.")
        recs = data.get("recommendations", [])

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

            if recs:
                st.markdown("### Recommended Assessments:")
                for r in recs:
                    st.markdown(f"- **{r['name']}** ({r['test_type']})")

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_reply}
        )

    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")