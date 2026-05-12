import streamlit as st
import json
from engine import get_consultant_response

st.set_page_config(page_title="SHL Consultant", layout="wide")

st.title("💼 SHL Solutions Consultant")
st.info("I help you design a tailored assessment strategy based on SHL's global catalog.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Explain your hiring needs..."):
    st.chat_message("user").markdown(prompt)
    
    # Build history with clear labels for the AI to 'remember' previous turns
    history_str = ""
    for msg in st.session_state.messages:
        role = "Consultant" if msg["role"] == "assistant" else "User"
        history_str += f"{role}: {msg['content']}\n"
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_raw_output = ""
        
        # STREAMING PHASE
        for chunk in get_consultant_response(prompt, history_str):
            full_raw_output += chunk
            # If using a thinking model, this masks the <thought> tags
            display = full_raw_output
            if "</thought>" in display: display = display.split("</thought>")[-1]
            response_placeholder.markdown(display + "▌")

 
        # PARSING PHASE
        try:
            # 1. Clean the output: Remove everything inside <thought> tags
            clean_output = full_raw_output
            if "</thought>" in clean_output:
                clean_output = clean_output.split("</thought>")[-1]
            elif "<thought>" in clean_output:
                # If it started thinking but never finished, don't show the thought
                clean_output = "Consultant is finalizing the response..."

            # 2. Try to find JSON inside the string (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', clean_output, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                ai_reply = data.get("reply", clean_output)
                recs = data.get("recommendations", [])
            else:
                # Fallback if no JSON structure found
                ai_reply = clean_output
                recs = []
            
            # 3. Final Render
            response_placeholder.markdown(ai_reply)
            
            if recs:
                st.write("---")
                # Using a container to prevent layout shifting
                with st.container():
                    cols = st.columns(len(recs) if 0 < len(recs) <= 3 else 3)
                    for idx, r in enumerate(recs):
                        with cols[idx % 3]:
                            st.success(f"**{r['name']}**")
                            st.caption(f"Type: {r['test_type']}")
                            st.markdown(f"[View in Catalog]({r['url']})")

            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        except Exception as e:
            # If everything fails, show the raw text so the screen isn't blank
            st.warning("Consultant response was not in the expected format, but here is the raw output:")
            response_placeholder.markdown(full_raw_output)
            st.session_state.messages.append({"role": "assistant", "content": full_raw_output})