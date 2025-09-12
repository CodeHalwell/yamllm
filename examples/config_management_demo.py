#!/usr/bin/env python3
"""
Configuration Management Demo

This example demonstrates the enhanced configuration management features:
- Configuration templates and presets
- Interactive configuration creation
- Configuration validation
- Provider-specific templates

Usage: uv run python examples/config_management_demo.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

def run_command(cmd_list, description, capture=True):
    """Run a command and optionally display results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd_list)}")
    print('='*60)
    
    try:
        if capture:
            result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=15)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"stderr: {result.stderr}")
            return result.returncode == 0
        else:
            return subprocess.run(cmd_list, timeout=15).returncode == 0
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Demonstrate configuration management features."""
    print("🎯 Configuration Management Demo")
    print("This demo shows the enhanced config features")
    
    # Check if we're in the right directory
    if not Path("yamllm").exists():
        print("❌ Please run this from the yamllm project root directory")
        sys.exit(1)
    
    # Create temporary directory for demo configs
    temp_dir = Path(tempfile.mkdtemp(prefix="yamllm_config_demo_"))
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # 1. List available presets
        run_command(["uv", "run", "python", "-m", "yamllm.cli", "config", "presets"], 
                   "Available Configuration Presets")
        
        # 2. List available models
        run_command(["uv", "run", "python", "-m", "yamllm.cli", "config", "models"], 
                   "Available Models")
        
        # 3. Create configurations with different presets
        configs_to_create = [
            {
                "name": "casual_openai",
                "provider": "openai", 
                "preset": "casual",
                "description": "Casual conversation with OpenAI"
            },
            {
                "name": "coding_anthropic",
                "provider": "anthropic",
                "preset": "coding", 
                "description": "Coding assistant with Anthropic"
            },
            {
                "name": "research_google",
                "provider": "google",
                "preset": "research",
                "description": "Research assistant with Google"
            }
        ]
        
        for config in configs_to_create:
            output_path = temp_dir / f"{config['name']}.yaml"
            success = run_command([
                "uv", "run", "python", "-m", "yamllm.cli", "config", "create",
                "--provider", config["provider"],
                "--preset", config["preset"],
                "--output", str(output_path),
                "--no-streaming"  # Disable streaming for demo
            ], f"Create {config['description']} Config")
            
            if success and output_path.exists():
                # 4. Validate the created configuration
                run_command([
                    "uv", "run", "python", "-m", "yamllm.cli", "config", "validate",
                    str(output_path)
                ], f"Validate {config['name']} Config")
                
                # Show a snippet of the generated config
                print(f"\n📄 Generated config preview ({config['name']}):")
                content = output_path.read_text()
                lines = content.split('\n')[:15]  # First 15 lines
                for line in lines:
                    print(f"  {line}")
                print("  ...")
        
        print(f"\n{'='*60}")
        print("🎉 Configuration Management Demo Complete!")
        print("\n💡 Key Features Demonstrated:")
        print("• Multiple configuration presets (casual, coding, research)")
        print("• Provider-specific templates (OpenAI, Anthropic, Google)")
        print("• Automatic model and tool selection")
        print("• Configuration validation with helpful error messages")
        print("• Template-based configuration generation")
        
        print(f"\n📁 Demo configs created in: {temp_dir}")
        print("\n🚀 Try creating your own config:")
        print("yamllm config create --provider openai --preset coding --output my_config.yaml")
        print('='*60)
        
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temp directory: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    main()