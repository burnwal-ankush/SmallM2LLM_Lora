"""
=============================================================================
app.py — ChatGPT-Style Chat UI (Gradio 6.x)
=============================================================================

A clean, minimal chat interface inspired by ChatGPT's design:
  - Dark sidebar with conversation history
  - Centered chat area with clean message bubbles
  - Bottom-anchored input bar
  - Model selector and settings in sidebar
  - Streaming responses

Usage:
    python3 app.py
    python3 app.py --model_dir smol-finetuned
    python3 app.py --share

=============================================================================
"""

import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, TextIteratorStreamer
from peft import AutoPeftModelForCausalLM
from threading import Thread

# =============================================================================
# ChatGPT-inspired CSS — Dark mode, clean, minimal
# =============================================================================

CSS = """
/* Global dark theme */
.gradio-container {
    background-color: #343541 !important;
    font-family: 'Söhne', 'ui-sans-serif', system-ui, -apple-system, sans-serif !important;
}

/* Sidebar */
.sidebar {
    background-color: #202123 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 0 !important;
    padding: 12px !important;
    min-height: 100vh;
}
.sidebar-title {
    color: white;
    font-size: 0.85em;
    font-weight: 600;
    padding: 10px 12px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.new-chat-btn {
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
    background: transparent !important;
    font-size: 0.9em !important;
    padding: 10px 12px !important;
    margin-bottom: 12px !important;
    width: 100% !important;
    text-align: left !important;
}
.new-chat-btn:hover {
    background: rgba(255,255,255,0.05) !important;
}

/* Chat area */
.chat-area {
    background-color: #343541 !important;
    border: none !important;
}
.chatbot {
    background-color: #343541 !important;
    border: none !important;
    min-height: 70vh;
}

/* Input area — bottom bar style */
.input-container {
    background: #40414f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin: 0 auto !important;
    max-width: 768px !important;
}
.msg-input textarea {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-size: 1em !important;
    padding: 12px 16px !important;
    resize: none !important;
}
.msg-input textarea::placeholder {
    color: rgba(255,255,255,0.4) !important;
}
.msg-input textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
.send-btn {
    background: #19c37d !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    min-width: 60px !important;
}
.send-btn:hover {
    background: #1a9d6a !important;
}
.send-btn:disabled {
    background: rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.3) !important;
}

/* Settings in sidebar */
.setting-label {
    color: rgba(255,255,255,0.7);
    font-size: 0.8em;
    margin-top: 12px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: rgba(25, 195, 125, 0.15);
    border: 1px solid rgba(25, 195, 125, 0.3);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78em;
    color: #19c37d;
    margin-top: 8px;
}

/* Footer text */
.footer-text {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.75em;
    padding: 8px;
    margin-top: 8px;
}

/* Hide default gradio footer */
footer { display: none !important; }

/* Slider styling */
.gr-slider input[type="range"] { accent-color: #19c37d !important; }
span.gr-slider { background: transparent !important; }
"""


# =============================================================================
# Model Loading
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_dir: str):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from {model_dir}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map=None
    )
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Model loaded!")
    return model, tokenizer, device


# =============================================================================
# App Builder
# =============================================================================

def create_app(model, tokenizer, device):

    with gr.Blocks(title="MyGPT") as demo:

        with gr.Row(equal_height=True):
            # =================================================================
            # Left Sidebar
            # =================================================================
            with gr.Column(scale=1, min_width=220, elem_classes=["sidebar"]):
                # New chat button
                new_chat_btn = gr.Button("+ New chat", elem_classes=["new-chat-btn"])

                gr.HTML('<div class="sidebar-title">⚙️ Settings</div>')

                # Temperature slider
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.1,
                    label="Temperature",
                )
                # Max tokens slider
                max_tokens = gr.Slider(
                    64, 1024, value=128, step=64,
                    label="Max tokens",
                )
                # Top-p slider
                top_p = gr.Slider(
                    0.1, 1.0, value=0.9, step=0.05,
                    label="Top-p",
                )

                # Model info
                gr.HTML(f"""
                    <div style="margin-top: 20px;">
                        <div class="sidebar-title">Model</div>
                        <div class="model-badge">SmolLM2-1.7B + LoRA</div>
                        <div style="color: rgba(255,255,255,0.5); font-size:0.78em; margin-top:8px;">
                            Running on {device.upper()}<br>
                            Fine-tuned on OpenHermes 2.5
                        </div>
                    </div>
                """)

                # Keyboard shortcuts info
                gr.HTML("""
                    <div class="footer-text" style="margin-top: auto; padding-top: 20px;">
                        Enter to send · Shift+Enter for new line
                    </div>
                """)

            # =================================================================
            # Main Chat Area
            # =================================================================
            with gr.Column(scale=4, elem_classes=["chat-area"]):
                # Chat display
                chatbot = gr.Chatbot(
                    label="",
                    elem_classes=["chatbot"],
                    placeholder="How can I help you today?",
                )

                # Input bar at the bottom
                with gr.Row(elem_classes=["input-container"]):
                    msg = gr.Textbox(
                        placeholder="Message MyGPT...",
                        show_label=False,
                        scale=6,
                        container=False,
                        elem_classes=["msg-input"],
                        lines=1,
                        max_lines=5,
                    )
                    send_btn = gr.Button("➤", elem_classes=["send-btn"], scale=0, min_width=50)

                # Footer disclaimer
                gr.HTML("""
                    <div class="footer-text">
                        MyGPT can make mistakes. Trained locally on Mac with LoRA fine-tuning.
                    </div>
                """)

        # =====================================================================
        # Chat Logic
        # =====================================================================

        def user_message(message, history):
            """Add user message to history and clear input."""
            if not message.strip():
                return "", history
            history = history or []
            history.append(gr.ChatMessage(role="user", content=message))
            return "", history

        def bot_response(history, temp_val, max_tok_val, top_p_val):
            """Generate streaming response from the model."""
            if not history:
                yield history
                return

            user_msg = history[-1].content if hasattr(history[-1], 'content') else history[-1]["content"]

            # Format with instruction template
            formatted = f"### Instruction:\nYou are a friendly, helpful AI assistant called MyGPT. Respond naturally and conversationally. If the user greets you, greet them back warmly.\n\n{user_msg}\n\n### Response:\n"
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            # Streaming generation
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_tok_val,
                temperature=temp_val,
                top_p=top_p_val,
                do_sample=True,
                repetition_penalty=1.15,
                streamer=streamer,
            )

            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()

            history.append(gr.ChatMessage(role="assistant", content=""))
            response = ""
            for token in streamer:
                if "### Instruction:" in response + token:
                    break
                response += token
                history[-1] = gr.ChatMessage(role="assistant", content=response.strip())
                yield history
            thread.join()

        def retry_last(history, temp_val, max_tok_val, top_p_val):
            """Remove last response and regenerate."""
            if history and len(history) >= 2:
                history = history[:-1]
                yield from bot_response(history, temp_val, max_tok_val, top_p_val)

        # =====================================================================
        # Event Wiring
        # =====================================================================

        # Send message on Enter or button click
        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, temperature, max_tokens, top_p], chatbot
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, temperature, max_tokens, top_p], chatbot
        )

        # New chat button clears everything
        new_chat_btn.click(lambda: [], outputs=[chatbot])

    return demo


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="smol-finetuned")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)
    demo = create_app(model, tokenizer, device)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
