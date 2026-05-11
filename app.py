import streamlit as st
import requests

st.set_page_config(page_title="SHL Consultant", layout="centered")

st.title("💼 SHL Solutions Consultant")
st.markdown("Enter your hiring needs below to get a tailored assessment stack.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help with your talent audit?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare payload for your FastAPI backend
    payload = {"messages": st.session_state.messages}
    
    try:
        # Call your local FastAPI server
        response = requests.post("http://127.0.0.1:8000/chat", json=payload)
        data = response.json()
        
        ai_reply = data.get("reply", "No reply received.")
        recs = data.get("recommendations", [])

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
            if recs:
                st.markdown("### Recommended Assessments:")
                for r in recs:
                    st.markdown(f"- **[{r['name']}]({r['url']})** ({r['test_type']})")

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")