# Built-in Tools

This page lists the built-in tools available in YAMLLM and their purpose.

Core utilities
- calculator: Safe math evaluation (+, -, *, /, **, %, basic functions)
- datetime: Current time or add offsets; optional formatting
- timezone: Convert time between timezones
- unit_converter: Common length, weight, temperature conversions
- random_string: Random alphanumeric string
- random_number: Random integer in a range
- uuid: Generate UUID v4 strings
- json_tool: Pretty/minify/validate JSON
- regex_extract: Extract regex matches with optional flags i/m/s
- lorem_ipsum: Generate placeholder text

Web + content
- web_search: DuckDuckGo search (no API key required)
- web_scraper: Fetch page, extract visible text
- url_metadata: Fetch page title and meta description
- weather: OpenWeatherMap current weather (requires WEATHER_API_KEY)

Files + data
- file_read: Read a small text file from the workspace (path and size restricted)
- file_search: Search workspace files by glob pattern
- csv_preview: Show CSV headers and first N rows

Encoding + crypto
- base64_encode: Base64-encode text
- base64_decode: Base64-decode text
- hash_text: Hash text with a selected algorithm (e.g., sha256)

Tool packs
- common: calculator, datetime, uuid, random_string, json_tool, regex_extract, lorem_ipsum
- web: web_search, web_scraper, url_metadata, weather
- files: file_read, file_search, csv_preview
- crypto: hash_text, base64_encode, base64_decode
- numbers: random_number, unit_converter
- time: datetime, timezone
- dev: json_tool, regex_extract, hash_text, base64_encode, base64_decode, file_read, file_search, csv_preview, uuid, random_string
- all: every tool

Introspection tool
- tools_help: List available tools (and optionally schemas). Example call:
  - names: optional list of tool names to filter
  - include_schema: boolean (default true) to include JSON Schema

Example YAML
```yaml
tools:
  enabled: true
  tool_timeout: 10
  packs: ["common", "web"]
  tool_list: ["unit_converter", "tools_help"]
```

