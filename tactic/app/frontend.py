# -*- coding: utf-8 -*-
import json
import requests
import gradio as gr
from loguru import logger
from tactic.utils import setup_logger
from tactic.app.server import BACKEND_SERVER_PORT, model_name


configs = {'log_dir': 'logs'}
setup_logger(configs)

# Set Web service parameters
frontend_service_port = 9002
SERVICE_URL = f"http://localhost:{BACKEND_SERVER_PORT}/v1/chat/completions"

event_dict = {
    'DraftAgent': '--- DraftAgent ---\n',
    'RefinementAgent_Analysis': '\n\n--- RefinementAgent ---\n[Analysis]: ',
    'RefinementAgent_Translation': '\n[Translation]: ',
    'Final_Translation': '\n[Final Translation]:\n'
}

def call_translation_stream(
    service_type,
    lang_pair,
    input_text,
    temperature,
    max_tokens
):
    logger.info(f"Stream request for service_type: {service_type}, lang_pair: {lang_pair}, "
                f"input_text: '{input_text[:50]}...', temperature: {temperature}, max_tokens: {max_tokens}")

    if not input_text:
        yield "The input text cannot be empty.", "" 
        return

    url = SERVICE_URL

    request_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{lang_pair}:\n{input_text}"}
        ]
    }

    logger.info(f"Sending stream request to URL: {url}")

    result_buffer = ""
    last_final_translation_content = ""

    try:
        response = requests.post(url, json=payload, headers=request_headers, stream=True, timeout=(5, 300))
        logger.info(f"Received stream response status: {response.status_code}")

        if response.status_code != 200:
            error_msg = f"The server returns a non-200 status code: {response.status_code}, details: {response.text}"
            logger.error(error_msg)
            yield error_msg, ""
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            logger.debug(f"Raw SSE Line: {line}")
            if line.strip() == "data: [DONE]":
                logger.info("Received [DONE] signal for stream.")
                break
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "")
                    if 'choices' in data and data['choices']:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        seg_prefix = ""
                        # Judge the event_type to determine whether to insert the value of event_dict
                        if event_type in event_dict and event_dict[event_type]:
                            seg_prefix = event_dict[event_type]
                        if content:
                            # Tags should only be added when inserting new paragraph content (to avoid repeating tags for each character).
                            if seg_prefix and not result_buffer.endswith(seg_prefix):
                                result_buffer += seg_prefix
                            result_buffer += content
                            if event_type == "Final_Translation":
                                last_final_translation_content = content.strip()
                        yield result_buffer, last_final_translation_content
                    else:
                        logger.warning(f"SSE event data missing 'choices' or empty: {data}")
                except Exception as e:
                    logger.error(f"Failed to parse SSE JSON or process: {e}, raw: {line}")
                    continue

        # The last wrap up ensures that the final text has a value
        yield result_buffer, last_final_translation_content

    except Exception as e:
        logger.error(f"General error during stream request setup: {e}")
        yield f"Error: Exception in the streaming request setting ({e})", ""

def call_translation_nostream(
    service_type,
    lang_pair,
    input_text,
    temperature,
    max_tokens
):
    """
    Invoke the translation backend service and handle non-streaming responses.
    Return two values: the complete translation process and the separate final translation.
    """
    logger.info(f"Non-stream request for service_type: {service_type}, lang_pair: {lang_pair}, "
                f"input_text: '{input_text[:50]}...', temperature: {temperature}, max_tokens: {max_tokens}")

    if not input_text:
        return "The input text cannot be empty.", ""

    if service_type in ['TACTIC-Full']:
        url = f"{SERVICE_URL}/full"
    else:
        url = SERVICE_URL

    request_headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{lang_pair}:\n{input_text}"}
        ]
    }

    logger.info(f"Sending non-stream request to URL: {url}")

    try:
        response = requests.post(url, json=payload, headers=request_headers, timeout=(5, 60))
        logger.info(f"Received non-stream response status: {response.status_code}")

        if response.status_code != 200:
            error_msg = f"The server returns a non-200 status code: {response.status_code}, details: {response.text}"
            logger.error(error_msg)
            return error_msg, ""

        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse non-stream response as JSON: {e}. Raw text: '{response.text}'")
            return f"Error: JSON parsing failed ({e}), original response is not valid JSON or is incorrectly formatted.", ""

        # Check structure
        if "choices" in response_data and response_data["choices"] and \
           "message" in response_data["choices"][0] and \
           "content" in response_data["choices"][0]["message"]:
            logger.info("Successfully parsed non-stream JSON response.")
            full_output_text = response_data["choices"][0]["message"]["content"]

            final_translation_content = ""
            final_tag = "[Final Translation]:\n"
            start_index = full_output_text.rfind(final_tag)
            if start_index != -1:
                final_translation_content = full_output_text[start_index + len(final_tag):].strip()
            else:
                logger.warning(f"'{final_tag}' not found in non-stream result. Using full output as fallback for final translation.")
                parts = full_output_text.rsplit(":", 1)
                if len(parts) > 1:
                    final_translation_content = parts[1].strip()
                else:
                    final_translation_content = full_output_text.strip()

            return full_output_text, final_translation_content
        else:
            error_msg = f"Response missing 'choices', 'message', or 'content' fields. Full response: {response_data}"
            logger.error(error_msg)
            return error_msg, ""

    except Exception as e:
        logger.error(f"General error for non-stream request: {e}")
        return f"Error: Request exception ({e})", ""

def call_translation_service(
    service_type,
    source_lang,
    target_lang,
    input_text,
    use_stream,
    temperature,
    max_tokens
):
    lang_pair = f"{source_lang.strip()}-{target_lang.strip()}"

    if use_stream:
        for process_text, final_text in call_translation_stream(service_type, lang_pair, input_text, temperature, max_tokens):
            yield update_combined_output(process_text, final_text), final_text
    else:
        process_text, final_text = call_translation_nostream(service_type, lang_pair, input_text, temperature, max_tokens)
        yield update_combined_output(process_text, final_text), final_text


with gr.Blocks(title="TACTIC Multi-Agent Translation", css=".main-center") as demo:
    with gr.Column(elem_classes=["main-center"]):
        gr.Markdown("""
            <div style='display:flex;justify-content:center;align-items:center;gap:14px;margin-bottom:0;'>
                <span style='font-size:2.3em;'>🌐</span>
                <span style='font-size:2.2em;font-weight:700;'>TACTIC Multi-Agent Translation</span>
            </div>
            <hr style='margin-top:4px;margin-bottom:8px;'>
        """)
    
        with gr.Row(equal_height=True, elem_classes=["main-content-row"]):
            with gr.Column(scale=2):
                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=["zh", "en", "ja", "fr", "de", "es", "ru"],
                        label="Source Language", 
                        value="zh", 
                        interactive=True, 
                        scale=1, 
                        elem_classes=["row-dropdown"]
                    )
                    target_lang = gr.Dropdown(
                        choices=["zh", "en", "ja", "fr", "de", "es", "ru"],
                        label="Target Language", 
                        value="en", 
                        interactive=True, 
                        scale=1, 
                        elem_classes=["row-dropdown"]
                    )
    
            with gr.Column(scale=1):
                service_type = gr.Dropdown(
                    choices=["TACTIC-Lite"],
                    label="Service Type", 
                    value="TACTIC-Lite", 
                    interactive=True
                )
                    
            with gr.Column(scale=1, min_width=180):
                with gr.Accordion("⚙️ Advanced settings", open=False):
                    use_stream = gr.Checkbox(label="Stream output", value=True, interactive=True)
                    temperature_number = gr.Number(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=0.6, 
                        interactive=True, 
                        label="Temperature"
                    )
                    max_tokens_number = gr.Number(
                        minimum=1, 
                        value=2048, 
                        precision=0, 
                        interactive=True, 
                        label="Max Tokens"
                    )
                
        with gr.Row(equal_height=True, elem_classes=["main-content-row"]):
            gr.HTML("")
            translate_btn = gr.Button("🚀 Translation", variant="primary", size="sm", min_width=180, elem_classes=["center-btn"])
            copy_btn = gr.Button("📋 Copy the final translation", variant="secondary", size="sm", min_width=180, elem_classes=["center-btn"])
            gr.HTML("")
            
        with gr.Row(equal_height=True, elem_classes=["main-content-row"]):
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    show_label=False,
                    placeholder="Please enter the content to be translated...",
                    lines=12, 
                    max_lines=20, 
                    interactive=True, 
                    autofocus=True,
                    elem_id="input-box"
                )
            with gr.Column(scale=1):
                combined_output = gr.HTML(
                    value="<div id='output-container'></div>",
                    elem_id="output-container"
                )
                hidden_copy_text = gr.Textbox(visible=False)
    
        # CSS 
        demo.css = """
        .row-dropdown .gr-dropdown-label, .row-dropdown .gr-dropdown {
            min-height: 42px !important;
            font-size: 1.04em !important;
            padding: 4px 8px !important;
            margin-bottom: 0 !important;
        }
        .gr-button, .center-btn {
            min-height: 42px !important;
            font-size: 1.08em !important;
            padding: 0 16px !important;
            margin-bottom: 0 !important;
            border-radius: 6px !important;
        }
        .center-btn {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 8px auto 0 auto !important;
            font-weight: bold;
            font-size: 1.12em;
        }
        #input-box {
            border: 1.5px solid #d1d5db !important;
            border-radius: 8px !important;
            background: #fff !important;
            box-shadow: 0 1px 2px rgba(60,80,120,.04) !important;
            min-height: 420px !important;
            width: 100% !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }
        
        #input-box textarea,
        #input-box textarea:focus,
        #input-box textarea:active,
        #input-box textarea:hover {
            background: #fff !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            filter: none !important;
            resize: none !important;
            padding: 18px 16px !important;
            font-size: 1em !important;
            color: #333 !important;
            transition: none !important;
        }
        .dark #input-box,
        .dark #input-box textarea,
        .dark #input-box textarea:focus,
        .dark #input-box textarea:active,
        .dark #input-box textarea:hover {
            background: #222533 !important;
            color: #fff !important;
            border-color: #44485c !important;
        }
    
        #output-container {
            border: 1.5px solid #d1d5db !important;
            border-radius: 8px !important;
            background: #fff !important;
            box-shadow: 0 1px 2px rgba(60,80,120,.04);
            min-height: 420px !important;
            width: 100% !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }
        
        #output-container *,
        #output-container *:before,
        #output-container *:after {
            border: none !important;
            box-shadow: none !important;
            background: none !important;
        }
        
        .process-text, .final-text {
            padding: 18px 16px !important;
            color: #555 !important;
            font-size: 0.95em !important;
        }
        
        .final-text {
            font-weight: 500 !important;
            color: #000 !important;
            font-size: 1.1em !important;
            margin-top: 10px !important;
        }
        
        .dark #output-container {
            border: 1.5px solid #44485c !important;
            background: #222533 !important;
        }
        .dark #output-container *, 
        .dark #output-container *:before, 
        .dark #output-container *:after {
            border: none !important;
            box-shadow: none !important;
            background: none !important;
        }
        .dark .final-text, .dark .process-text {
            color: #fff !important;
        }
        
        hr.divider-line,
        #output-container hr,
        #output-container .divider-line {
            border: none;
            border-top: 1px solid #e0e0e0 !important;
            margin: 18px 0 10px 0 !important;
            height: 0 !important;
            background: none !important;
            display: block !important;
            opacity: 1 !important;
        }
        .dark hr.divider-line,
        .dark #output-container hr.divider-line {
            border-top: 1px solid #44485c !important;
        }
        
        .main-center {
            max-width: 1000px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            width: 100% !important;
        }
        .main-center {
            max-width: 1120px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            width: 100% !important;
        }
        
        .main-content-row {
            width: 100% !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: stretch !important;
        }
        .main-content-row > .gr-column {
            width: 50% !important;
            min-width: 300px !important;
            max-width: 700px !important;
        }
        
        #input-box, #output-container {
            width: 95% !important;
            margin: 0 auto !important;
            background: #fff !important;
        }
        """
    
        def update_combined_output(process_text, final_text):
            def format_text(text):
                if not text:
                    return ""
                return text.replace("\n", "<br>")
            if not final_text:
                return f"""
                <div class="process-text">{format_text(process_text)}</div>
                """
            return f"""
            <div class="process-text">{format_text(process_text)}</div>
            <hr class="divider-line">
            <div class="final-text">{format_text(final_text)}</div>
            """
    
    
        def copy_final_text(final_text):
            return final_text
    
        translation_inputs = [
            service_type, source_lang, target_lang, input_text,
            use_stream, temperature_number, max_tokens_number
        ]
        translation_outputs = [combined_output, hidden_copy_text]
    
        translate_btn.click(
            fn=call_translation_service,
            inputs=translation_inputs,
            outputs=translation_outputs
        )
        input_text.submit(
            fn=call_translation_service,
            inputs=translation_inputs,
            outputs=translation_outputs
        )
        copy_btn.click(
            fn=None,
            inputs=hidden_copy_text,
            outputs=None,
            js="""
            function(text) {
                navigator.clipboard.writeText(text);
                setTimeout(() => alert('The final translation has been copied to the clipboard'), 100);
                return [];
            }
            """
        )

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    demo.launch(server_name="0.0.0.0", server_port=frontend_service_port)
