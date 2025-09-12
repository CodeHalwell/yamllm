"""
Enhanced chat interface for YAMLLM with beautiful Rich UI.

This module provides a complete chat interface with streaming support,
tool visualization, and session management.
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import json

from yamllm.core.llm import LLM
from yamllm.ui.components import YAMLLMConsole, StreamingDisplay, ToolExecutionDisplay
from yamllm.core.parser import parse_yaml_config


class ChatInterface:
    """Beautiful chat interface for YAMLLM."""
    
    def __init__(self, config_path: str, api_key: str, theme: str = "default"):
        """Initialize chat interface."""
        self.config_path = config_path
        self.api_key = api_key
        self.console = YAMLLMConsole(theme=theme)
        self.streaming_display = StreamingDisplay(self.console)
        self.tool_display = ToolExecutionDisplay(self.console)
        
        # Initialize LLM
        self.llm = LLM(config_path, api_key)
        
        # Load config for display
        self.config = parse_yaml_config(config_path)
        
        # Session tracking
        self.message_count = 0
        self.tool_call_count = 0
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Set up LLM callbacks for UI updates."""
        # Streaming callback
        def on_stream(text: str):
            self.streaming_display.update(text)
        
        self.llm.set_stream_callback(on_stream)
        
        # Event callback for tools and other events
        def on_event(event: Dict[str, Any]):
            event_type = event.get("type")
            
            if event_type == "tool_call":
                # Tool is being called
                self.tool_call_count += 1
                tool_name = event.get("name")
                args = event.get("args", {})
                
                # Show tool execution start
                self.console.console.print(
                    f"\n[{self.console.theme['tool']}]🔧 Calling {tool_name}...[/{self.console.theme['tool']}]"
                )
                
            elif event_type == "tool_result":
                # Tool execution completed
                tool_name = event.get("name")
                result = event.get("result")
                
                # Parse result if it's a string
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        pass
                
                # Show tool result (simplified)
                self.console.console.print(
                    f"[{self.console.theme['success']}]✓ {tool_name} completed[/{self.console.theme['success']}]"
                )
                
            elif event_type == "thinking":
                # Show thinking process
                step = event.get("step")
                content = event.get("content")
                
                if step and content:
                    self.console.console.print(
                        f"\n[{self.console.theme['thinking']}]💭 {step}: {content[:100]}...[/{self.console.theme['thinking']}]"
                    )
        
        self.llm.set_event_callback(on_event)
    
    def run(self):
        """Run the chat interface."""
        try:
            # Clear screen and show banner
            self.console.console.clear()
            self.console.print_banner()
            
            # Show configuration
            self.console.print_config_summary(self.config.dict())
            
            # Welcome message
            self.console.console.print(
                f"\n[{self.console.theme['info']}]Welcome to YAMLLM Chat! Type 'exit' to quit, 'help' for commands.[/{self.console.theme['info']}]\n"
            )
            
            # Main chat loop
            while True:
                try:
                    # Get user input
                    user_input = self.console.prompt_user()
                    
                    # Check for commands
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        if self.console.confirm("Are you sure you want to exit?"):
                            break
                        continue
                    
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    
                    elif user_input.lower() == 'stats':
                        self._show_stats()
                        continue
                    
                    elif user_input.lower() == 'clear':
                        self.console.console.clear()
                        self.console.print_banner()
                        continue
                    
                    elif user_input.lower() == 'config':
                        self.console.print_config_summary(self.config.dict())
                        continue
                    
                    # Process message
                    self.message_count += 1
                    
                    # Show user message
                    self.console.print_message(
                        role="user",
                        content=user_input,
                        timestamp=datetime.now()
                    )
                    
                    # Get LLM response
                    if self.llm.output_stream:
                        # Streaming response with cooperative cancellation
                        self.streaming_display.start("assistant")
                        try:
                            response = self.llm.query(user_input)
                        except KeyboardInterrupt:
                            try:
                                self.llm.cancel()
                            except Exception:
                                pass
                            self.console.console.print(
                                f"\n[{self.console.theme['warning']}]Interrupted. Streaming cancelled.[/{self.console.theme['warning']}]"
                            )
                            response = None
                        finally:
                            self.streaming_display.stop()
                    else:
                        # Non-streaming response
                        with self.console.live_status("🤔 Thinking..."):
                            response = self.llm.query(user_input)
                        
                        # Show response
                        self.console.print_message(
                            role="assistant",
                            content=response,
                            timestamp=datetime.now(),
                            tokens=len(response.split()) if response else 0
                        )
                    
                except KeyboardInterrupt:
                    self.console.console.print(
                        f"\n[{self.console.theme['warning']}]Interrupted. Type 'exit' to quit.[/{self.console.theme['warning']}]"
                    )
                    continue
                
                except Exception as e:
                    self.console.print_error(e)
                    continue
            
            # Show final stats
            self._show_stats()
            self.console.console.print(
                f"\n[{self.console.theme['info']}]Thank you for using YAMLLM! Goodbye! 👋[/{self.console.theme['info']}]\n"
            )
            
        finally:
            # Cleanup
            self.llm.cleanup()
    
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]help[/cyan]     - Show this help message
  [cyan]exit[/cyan]     - Exit the chat
  [cyan]clear[/cyan]    - Clear the screen
  [cyan]stats[/cyan]    - Show session statistics
  [cyan]config[/cyan]   - Show configuration

[bold]Tips:[/bold]
  • Press Ctrl+C to interrupt a response
  • The chat maintains conversation history if memory is enabled
  • Tools will be automatically used when appropriate
"""
        self.console.console.print(help_text)
    
    def _show_stats(self):
        """Show session statistics."""
        stats = {
            "messages": self.message_count,
            "tool_calls": self.tool_call_count
        }
        
        # Add token usage if available
        if hasattr(self.llm, '_total_usage'):
            stats.update(self.llm._total_usage)
            stats["model"] = self.llm.model
        
        self.console.print_stats(stats)


class AsyncChatInterface(ChatInterface):
    """Async version of the chat interface for better performance."""
    
    def __init__(self, config_path: str, api_key: str, theme: str = "default"):
        """Initialize async chat interface."""
        super().__init__(config_path, api_key, theme)
        self.tasks = []
    
    async def arun(self):
        """Run the async chat interface."""
        try:
            # Clear screen and show banner
            self.console.console.clear()
            self.console.print_banner()
            
            # Show configuration
            self.console.print_config_summary(self.config.dict())
            
            # Welcome message
            self.console.console.print(
                f"\n[{self.console.theme['info']}]Welcome to YAMLLM Async Chat! Type 'exit' to quit.[/{self.console.theme['info']}]\n"
            )
            
            # Main chat loop
            while True:
                try:
                    # Get user input (in thread to not block)
                    loop = asyncio.get_event_loop()
                    user_input = await loop.run_in_executor(
                        None, self.console.prompt_user
                    )
                    
                    # Check for exit
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        break
                    
                    # Process message
                    self.message_count += 1
                    
                    # Show user message
                    self.console.print_message(
                        role="user",
                        content=user_input,
                        timestamp=datetime.now()
                    )
                    
                    # Get async LLM response
                    with self.console.live_status("🤔 Processing..."):
                        response = await self.llm.aquery(user_input)
                    
                    # Show response
                    self.console.print_message(
                        role="assistant",
                        content=response,
                        timestamp=datetime.now(),
                        tokens=len(response.split()) if response else 0
                    )
                    
                except KeyboardInterrupt:
                    self.console.console.print(
                        f"\n[{self.console.theme['warning']}]Interrupted.[/{self.console.theme['warning']}]"
                    )
                    break
                
                except Exception as e:
                    self.console.print_error(e)
                    continue
            
            # Show final stats
            self._show_stats()
            self.console.console.print(
                f"\n[{self.console.theme['info']}]Goodbye! 👋[/{self.console.theme['info']}]\n"
            )
            
        finally:
            # Cleanup
            self.llm.cleanup()
    
    def run(self):
        """Run the async interface."""
        asyncio.run(self.arun())


def create_chat_interface(
    config_path: str,
    api_key: Optional[str] = None,
    theme: str = "default",
    async_mode: bool = False
) -> ChatInterface:
    """
    Create a chat interface with the specified configuration.
    
    Args:
        config_path: Path to YAML configuration
        api_key: API key (will use environment variable if not provided)
        theme: UI theme (default, monokai, dracula)
        async_mode: Use async interface for better performance
    
    Returns:
        ChatInterface instance
    """
    import os
    
    # Get API key
    if not api_key:
        # Try to get from environment based on config
        config = parse_yaml_config(config_path)
        provider = config.provider.name.upper()
        api_key = os.getenv(f"{provider}_API_KEY")
        
        if not api_key:
            raise ValueError(
                f"No API key provided. Set {provider}_API_KEY environment variable."
            )
    
    # Create appropriate interface
    if async_mode:
        return AsyncChatInterface(config_path, api_key, theme)
    else:
        return ChatInterface(config_path, api_key, theme)
