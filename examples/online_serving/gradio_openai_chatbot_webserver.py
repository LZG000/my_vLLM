# SPDX-License-Identifier: Apache-2.0

import argparse

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(message, history):
    history_openai_format = [{
        "role": "system",
        "content": "You are a great ai assistant."
    }]

    # ----------------- 终极修复，兼容所有 Gradio 版本 -----------------
    for turn in history:
        try:
            # 情况1：元组 / 列表 (user, assistant)
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_msg, ai_msg = turn[0], turn[1]
            # 情况2：字典格式
            elif isinstance(turn, dict):
                user_msg = turn.get("user", turn.get("human", ""))
                ai_msg = turn.get("assistant", turn.get("bot", ""))
            # 情况3：其他格式直接跳过
            else:
                continue

            if user_msg and ai_msg:
                history_openai_format.append({"role": "user", "content": user_msg})
                history_openai_format.append({"role": "assistant", "content": ai_msg})
        except:
            continue

    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,
        messages=history_openai_format,
        temperature=args.temp,
        stream=True,
        extra_body={
            'repetition_penalty': 1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',')
                if id.strip()
            ] if args.stop_token_ids else []
        })

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        partial_message += content
        yield partial_message


# Create and launch a chat interface with Gradio
gr.ChatInterface(predict).queue().launch(server_name=args.host,
                                         server_port=args.port,
                                         share=True)