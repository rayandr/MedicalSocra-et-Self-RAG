import gradio as gr
from gradio.themes.default import Default
from self_rag_v6 import app  # your backend module



# --- Backend function: returns OpenAI-style message ---
def respond(message, chat_history):
    final_output = None
    for output in app.stream({"question": message}):
        for key, value in output.items():
            if key == "generate":
                final_output = value["generation"]
    return {"role": "assistant", "content": final_output}



# --- Custom theme using valid token names ---
custom_theme = Default(
    primary_hue="teal",
    secondary_hue="gray",
    neutral_hue="gray",
    font=["Inter", "sans-serif"]
).set(
    button_primary_background_fill="#14b8a6",   # teal-600
    button_primary_text_color="#ffffff",        # white
    button_secondary_background_fill="#e5e7eb", # gray-200
    button_secondary_text_color="#000000",      # black
    body_background_fill="#f9fafb",             # gray-50
    border_color_primary="#2dd4bf",             # teal-400
    color_accent_soft="#99f6e4",                # teal-100
    link_text_color="#0f766e"                   # teal-700
)




# --- Dark mode CSS overrides ---
custom_css = """
.dark .gr-button {
    background-color: #14b8a6 !important;
    color: white !important;
}
.dark .gr-chat-message-user {
    background-color: #0f766e !important;
    color: white !important;
}
.dark .gr-chat-message-assistant {
    background-color: #1e293b !important;
    color: white !important;
}
"""




# --- Build chat interface inside styled Blocks ---
with gr.Blocks(theme=custom_theme, css=custom_css) as chat_ui:
    gr.ChatInterface(
        fn=respond,
        title="Self-RAG Chatbot",
        description="Ask questions about diabetes. This AI routes your query to either a vector database of trusted sources or live web search using LangGraph.",
        examples=[
            "What are the symptoms of type 1 diabetes?",
            "What causes insulin resistance?",
            "Is type 2 diabetes reversible?",
        ],
        type="messages",  # ✅ avoids deprecation warning
        # submit_btn="💬 Send",
        # clear_btn="🧹 Clear",
        # retry_btn="🔁 Retry",
        # stop_btn="⏹️ Stop",
    )




# --- Launch ---
if __name__ == "__main__":
    chat_ui.launch()
