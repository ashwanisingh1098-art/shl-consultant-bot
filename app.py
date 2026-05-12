import streamlit as st
import requests

st.set_page_config(page_title="SHL Consultant", layout="centered")

st.title("💼 SHL Solutions Consultant")
st.markdown("Enter your hiring needs below to get a tailored assessment stack.")

API_URL = "https://shl-consultant-bot.onrender.com/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help with your talent audit?"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    payload = {"messages": st.session_state.messages}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error(f"Backend error: {response.status_code}")
        else:
            data = response.json()

            ai_reply = data.get("reply", "No reply received.")
            recs = data.get("recommendations", [])

            with st.chat_message("assistant"):
                st.markdown(ai_reply)

                if recs:
                    st.markdown("### Recommended Assessments:")
                    for r in recs:
                        st.markdown(f"- **[{r['name']}]({r['url']})** ({r['test_type']})")

            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    except Exception as e:
        st.error(f"Error connecting to backend: {e}")