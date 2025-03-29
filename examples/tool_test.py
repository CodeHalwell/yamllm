# test_duckduckgo.py
from yamllm.tools.utility_tools import WeatherTool
import json
import os

# Create WebSearch tool instance
search_tool = WeatherTool(api_key=os.environ.get('WEATHER_API_KEY'))

# Test the tool
results = search_tool.execute("Northwich")

print(json.dumps(results, indent=2))