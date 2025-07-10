import streamlit as st
from medical_socra_RAG_v3 import medical_socra_run  # Ensure this matches the actual function


st.title("Medical Socratic Assistant")
st.write("Enter symptoms below to start the diagnostic dialogue.")

# Text input
symptoms_input = st.text_area("Enter your symptoms:", height=150)

# Submit button
if st.button("Submit"):
    if symptoms_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        with st.spinner("Analyzing..."):
            result = medical_socra_run(symptoms_input)
            # result = symptoms_input
        # st.text_area("System Response", value=result, height=300)
        st.code(result, language="markdown")

        # st.markdown(
        #     f"""
        #     <div style='max-height: 600px; overflow-y: auto; padding: 1rem; border-radius: 8px;
        #                 background-color: #0e1117; color: white;
        #                 white-space: pre-wrap; word-break: break-word;
        #                 font-family: monospace; line-height: 1.6;'>
        #         {result}
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        # st.markdown(result)

