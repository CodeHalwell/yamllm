#!/usr/bin/env python3
"""
Simple YAMLLM UI Demo - 10 lines of beautiful UI

This shows the proper way to use YAMLLM's built-in UI components
for a beautiful experience in just a few lines of code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from yamllm.core.llm import LLM
from yamllm.ui.components import YAMLLMConsole, StreamingDisplay

def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY")
        return
    
    # Create beautiful UI console
    ui = YAMLLMConsole(theme="default")
    ui.print_banner()
    
    # Create LLM with simple config
    llm = LLM(
        config_path=".config_examples/openai/basic_config_openai.yaml",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Show config summary
    config = {"provider": {"name": "openai", "model": "gpt-4o-mini"}}
    ui.print_config_summary(config)
    
    # Simple chat loop with streaming
    stream = StreamingDisplay(ui)
    
    while True:
        prompt = ui.prompt_user("You")
        if not prompt or prompt.lower() in ["exit", "quit"]:
            break
            
        ui.print_message("user", prompt)
        stream.start("assistant")
        
        try:
            response = llm.get_response(prompt)
            ui.print_message("assistant", response)
        except Exception as e:
            ui.print_error(e)
        finally:
            stream.stop()

if __name__ == "__main__":
    main()