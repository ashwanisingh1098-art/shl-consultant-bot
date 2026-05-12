import streamlit as st
import json
from engine import get_consultant_response

st.set_page_config(page_title="SHL Consultant", layout="centered")

st.title("💼 SHL Solutions Consultant")
st.markdown("Enter your hiring needs below to get a tailored assessment stack.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help with your talent audit?"):

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build history string
    history_str = ""
    for msg in st.session_state.messages[:-1]:
        history_str += f"{msg['role']}: {msg['content']}\n"

    try:
        raw_output = get_consultant_response(prompt, history_str)

        try:
            data = json.loads(raw_output)
        except:
            data = {
                "reply": raw_output,
                "recommendations": [],
                "end_of_conversation": False
            }

        ai_reply = data.get("reply", "No reply received.")
        recs = data.get("recommendations", [])

        with st.chat_message("assistant"):
            st.markdown(ai_reply)

            if recs:
                st.markdown("### Recommended Assessments")
                for r in recs:
                    st.markdown(f"- **[{r['name']}]({r['url']})** ({r['test_type']})")

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    except Exception as e:
        st.error(f"Backend error: {e}")