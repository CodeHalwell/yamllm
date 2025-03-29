# test_duckduckgo.py
from yamllm.tools.utility_tools import WebSearch
import json

# Create WebSearch tool instance
search_tool = WebSearch()

# Test the tool
results = search_tool.execute(query="latest AI developments", max_results=3)

print(json.dumps(results, indent=2))