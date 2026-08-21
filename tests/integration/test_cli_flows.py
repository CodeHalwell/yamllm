"""
CLI integration tests.

Tests end-to-end CLI flows including tool visibility, error handling,
and config validation.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys


class TestCLIWithTools:
    """Test CLI chat functionality with tools."""
    
    def test_cli_chat_basic_invocation(self):
        """Test basic CLI chat invocation."""
        from yamllm.cli.main import main
        
        # Test with --help flag
        with patch.object(sys, 'argv', ['yamllm', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with 0
            assert exc_info.value.code == 0
    
    def test_cli_status_command(self):
        """Test CLI status command."""
        from yamllm.cli.main import show_status
        import argparse
        
        # Create mock args
        args = argparse.Namespace()
        
        # Should return 0 on success
        result = show_status(args)
        assert result == 0
    
    @patch('yamllm.cli.tools.Console')
    @patch('yamllm.cli.tools.parse_yaml_config')
    @patch('yamllm.core.llm.LLM')
    def test_cli_list_tools(self, mock_llm_class, mock_parse_config, mock_console):
        """Test CLI tool listing."""
        from yamllm.cli.tools import list_tools
        import argparse
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
provider:
  name: openai
  api_key: test
tools:
  enabled: true
  tool_list: [calculator, web_search]
""")
            config_path = f.name
        
        try:
            # Mock config parsing
            mock_config = MagicMock()
            mock_config.tools = MagicMock()
            mock_config.tools.enabled = True
            mock_config.tools.tool_list = ['calculator', 'web_search']
            mock_parse_config.return_value = mock_config
            
            # Create args
            args = argparse.Namespace(config=config_path)
            
            # Run list_tools
            result = list_tools(args)
            
            # Should succeed
            assert result == 0
            
        finally:
            os.unlink(config_path)


class TestCLIToolVisibility:
    """Test CLI tool visibility and registration."""
    
    @patch('yamllm.core.llm.LLM')
    @patch('yamllm.cli.tools.parse_yaml_config')
    def test_tool_visibility_in_config(self, mock_parse_config, mock_llm):
        """Test that tools are visible when configured."""
        from yamllm.cli.tools import list_tools
        import argparse
        
        # Mock configuration
        mock_config = MagicMock()
        mock_config.tools = MagicMock()
        mock_config.tools.enabled = True
        mock_config.tools.tool_list = ['calculator', 'web_search', 'datetime']
        mock_parse_config.return_value = mock_config
        
        # Create args
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            args = argparse.Namespace(config=config_path)
            
            # List tools should work
            result = list_tools(args)
            assert result == 0
            
        finally:
            os.unlink(config_path)
    
    @patch('yamllm.cli.tools.Console')
    @patch('yamllm.cli.tools.parse_yaml_config')
    def test_tool_packs_visibility(self, mock_parse_config, mock_console):
        """Test tool pack visibility."""
        from yamllm.cli.tools import list_available_packs
        import argparse
        
        args = argparse.Namespace()
        
        # Should list available packs
        result = list_available_packs(args)
        assert result == 0


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_cli_missing_config_file(self):
        """Test error handling for missing config file."""
        from yamllm.cli.chat import run_chat
        import argparse
        
        args = argparse.Namespace(
            config='nonexistent.yaml',
            message='test',
            stream=False,
            show_thinking=False
        )
        
        # Should handle missing file gracefully
        with pytest.raises((FileNotFoundError, SystemExit)):
            run_chat(args)
    
    @patch('yamllm.cli.config.Console')
    def test_cli_invalid_config_format(self, mock_console):
        """Test error handling for invalid config format."""
        from yamllm.cli.config import validate_config
        import argparse
        
        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            args = argparse.Namespace(config=config_path)
            
            # Should handle invalid YAML
            with pytest.raises(Exception):
                validate_config(args)
                
        finally:
            os.unlink(config_path)
    
    def test_cli_keyboard_interrupt(self):
        """Test handling of keyboard interrupt."""
        from yamllm.cli.main import main
        
        # Simulate Ctrl+C
        with patch('yamllm.cli.main.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.side_effect = KeyboardInterrupt()
            
            # Should exit gracefully
            with pytest.raises(SystemExit):
                main()


class TestCLIConfigValidation:
    """Test CLI config validation."""
    
    @patch('yamllm.cli.config.Console')
    @patch('yamllm.cli.config.parse_yaml_config')
    @patch('yamllm.cli.config.ConfigValidator')
    def test_validate_valid_config(self, mock_validator, mock_parse, mock_console):
        """Test validation of valid config."""
        from yamllm.cli.config import validate_config
        import argparse
        
        # Mock valid config
        mock_config = MagicMock()
        mock_parse.return_value = mock_config
        mock_validator.validate_config.return_value = []  # No errors
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("provider:\n  name: openai\n")
            config_path = f.name
        
        try:
            args = argparse.Namespace(config=config_path)
            result = validate_config(args)
            assert result == 0
            
        finally:
            os.unlink(config_path)
    
    @patch('yamllm.cli.config.Console')
    @patch('yamllm.cli.config.parse_yaml_config')
    @patch('yamllm.cli.config.ConfigValidator')
    def test_validate_invalid_config(self, mock_validator, mock_parse, mock_console):
        """Test validation of invalid config."""
        from yamllm.cli.config import validate_config
        import argparse
        
        # Mock invalid config
        mock_config = MagicMock()
        mock_parse.return_value = mock_config
        mock_validator.validate_config.return_value = ["Error 1", "Error 2"]  # Has errors
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("provider:\n  name: invalid\n")
            config_path = f.name
        
        try:
            args = argparse.Namespace(config=config_path)
            result = validate_config(args)
            # Should indicate failure
            assert result == 1
            
        finally:
            os.unlink(config_path)


class TestCLIMemoryCommands:
    """Test CLI memory management commands."""
    
    @patch('yamllm.cli.memory.Console')
    @patch('yamllm.cli.memory.ConversationStore')
    def test_list_conversations(self, mock_store_class, mock_console):
        """Test listing conversations."""
        from yamllm.cli.memory import list_conversations
        import argparse
        
        # Mock store
        mock_store = MagicMock()
        mock_store.list_conversations.return_value = [
            {'id': 1, 'title': 'Test Chat', 'created_at': '2024-01-01'}
        ]
        mock_store_class.return_value = mock_store
        
        args = argparse.Namespace(db_path=':memory:')
        result = list_conversations(args)
        
        assert result == 0
    
    @patch('yamllm.cli.memory.Console')
    @patch('yamllm.cli.memory.ConversationStore')
    def test_clear_memory(self, mock_store_class, mock_console):
        """Test clearing memory."""
        from yamllm.cli.memory import clear_memory
        import argparse
        
        # Mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        args = argparse.Namespace(
            db_path=':memory:',
            confirm=True
        )
        
        result = clear_memory(args)
        assert result == 0


class TestCLISetupWizard:
    """Test CLI setup wizard."""
    
    @patch('yamllm.core.setup_wizard.SetupWizard')
    @patch('yamllm.cli.main.Console')
    def test_setup_wizard_invocation(self, mock_console, mock_wizard_class):
        """Test invoking setup wizard."""
        from yamllm.cli.main import run_setup
        import argparse
        
        # Mock wizard
        mock_wizard = MagicMock()
        mock_wizard.run.return_value = True
        mock_wizard_class.return_value = mock_wizard
        
        args = argparse.Namespace()
        result = run_setup(args)
        
        # Wizard should be called
        mock_wizard.run.assert_called_once()
        assert result == 0


class TestCLIOutputFormatting:
    """Test CLI output formatting."""
    
    @patch('yamllm.cli.tools.Console')
    def test_tool_list_formatting(self, mock_console_class):
        """Test tool list output formatting."""
        from yamllm.cli.tools import list_available_packs
        import argparse
        
        # Mock console
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        args = argparse.Namespace()
        result = list_available_packs(args)
        
        # Should format output
        assert result == 0
        mock_console.print.assert_called()


class TestCLIVersionInfo:
    """Test CLI version information."""
    
    def test_version_display(self):
        """Test version display."""
        from yamllm.cli.main import __version__
        
        # Version should be defined
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestCLIHelpText:
    """Test CLI help text."""
    
    def test_main_help(self):
        """Test main CLI help text."""
        from yamllm.cli.main import main
        
        with patch.object(sys, 'argv', ['yamllm', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    def test_subcommand_help(self):
        """Test subcommand help text."""
        from yamllm.cli.main import main
        
        # Test tools subcommand help
        with patch.object(sys, 'argv', ['yamllm', 'tools', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
