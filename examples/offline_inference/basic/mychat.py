# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")

    # Create an LLM
    llm = LLM(**args)

    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # 可选：自定义聊天模板
    chat_template = None
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            chat_template = f.read()

    print("=" * 80)
    print("🎉 交互式聊天模式启动！输入 quit 或 exit 退出")
    print("=" * 80)

    # 初始化对话历史（保持多轮对话）
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # 无限循环对话
    while True:
        # 接收你的输入
        user_input = input("\n👤 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 结束对话！")
            break

        # 把用户输入加入历史
        conversation.append({"role": "user", "content": user_input})

        # 模型生成回答
        outputs = llm.chat(conversation, sampling_params, chat_template=chat_template, use_tqdm=False)
        response = outputs[0].outputs[0].text

        # 打印回答
        print(f"\n🤖 AI: {response}")
        print("-" * 80)

        # 把AI回答加入历史，实现多轮对话
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model="Qwen/Qwen2.5-1.5B-Instruct")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=1024)
    sampling_group.add_argument("--temperature", type=float, default=0.7)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add example params
    parser.add_argument("--chat-template-path", type=str)
    args: dict = vars(parser.parse_args())
    main(args)