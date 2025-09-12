from .base import Tool
from typing import List, Dict, Optional, Any
from datetime import datetime
import pytz
import json
from duckduckgo_search import DDGS
import requests
import os
import dotenv
from bs4 import BeautifulSoup
import math

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Import network base early to use in classes below
from .network_base import NetworkTool, NetworkError
from .security import SecurityManager, ToolExecutionError

class WeatherTool(NetworkTool):
    "Tool to get current weather information from OpenWeatherMap API. Query is performed by city name."
    def __init__(self, api_key: str, timeout: int = 15, max_retries: int = 3, security_manager: Optional[SecurityManager] = None):
        super().__init__(
            name="weather",
            description="Get current weather information from OpenWeatherMap API",
            timeout=timeout,
            max_retries=max_retries,
            security_manager=security_manager,
        )

        self.api_key = os.environ.get('WEATHER_API_KEY') if api_key is None else api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.params = {
            "appid": self.api_key,
            "units": "metric"  # Use metric units by default
        }

    def execute(self, location: str) -> Dict:
        """
        Execute a weather query using OpenWeatherMap API.
        """
        try:
            if not self.api_key:
                return {"error": "Missing WEATHER_API_KEY for weather tool"}
            self.params["q"] = location
            response = self.make_request("GET", self.base_url, params=self.params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("cod") != 200:
                return {"error": "City not found"}
            
            # Extract relevant information from the response
            weather_info = {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
            
            return weather_info
        except NetworkError as e:
            return {"error": f"Network error: {e}"}
        except Exception as e:
            return {"error": f"Weather fetch failed: {e}"}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'London'"}
            },
            "required": ["location"],
        }

class WebSearch(NetworkTool):
    """
    Tool to perform web searches using DuckDuckGo API. This performs a search query and returns the results, typically a wide range of information.
    """
    def __init__(self, api_key: str = None, timeout: int = 15, max_retries: int = 3, providers: Optional[List[Any]] = None, security_manager: Optional[SecurityManager] = None):  # DuckDuckGo no key
        super().__init__(
            name="web_search",
            description="Search the web for current information using DuckDuckGo/SerpAPI",
            timeout=timeout,
            max_retries=max_retries,
            security_manager=security_manager,
        )
        self.api_key = api_key  # Unused for DDG; kept for compatibility
        self.providers = providers or self._build_providers()

    def _build_providers(self) -> List[Any]:
        providers: List[Any] = [DuckDuckGoProvider()]
        
        # Add SerpAPI provider if key is available
        serp_key = os.environ.get("SERPAPI_API_KEY")
        if serp_key:
            providers.append(SerpAPIProvider(session=self.session, api_key=serp_key, timeout=self.timeout, max_retries=self.max_retries))
        
        # Add Tavily provider if key is available
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            providers.append(TavilyProvider(session=self.session, api_key=tavily_key, timeout=self.timeout, max_retries=self.max_retries))
        
        # Add Bing Search provider if key is available
        bing_key = os.environ.get("BING_SEARCH_API_KEY")
        if bing_key:
            providers.append(BingSearchProvider(session=self.session, api_key=bing_key, timeout=self.timeout, max_retries=self.max_retries))
        
        return providers

    def execute(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Execute a web search using DuckDuckGo.
        """
        last_error = None
        for provider in self.providers:
            try:
                results = provider.search(query, max_results=max_results)
                if results:
                    return {"query": query, "num_results": len(results), "results": results}
            except Exception as e:
                last_error = str(e)
                continue
        return {"error": f"All search providers failed. Last error: {last_error}"}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Limit number of results", "default": 5},
            },
            "required": ["query"],
        }


class DuckDuckGoProvider:
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        results: List[Dict[str, str]] = []
        attempts = 0
        last_err = None
        # Try a few times to ride out transient 202 rate limits
        while attempts < 4 and not results:
            attempts += 1
            try:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=max_results))
                    for result in search_results:
                        results.append({
                            "title": result.get("title", "No title"),
                            "snippet": result.get("body", "No description"),
                            "url": result.get("href", "No URL"),
                        })
            except Exception as e:
                last_err = str(e)
                # Exponential backoff on known 202 rate limit
                if "202 Ratelimit" in (last_err or ""):
                    import time as _t
                    _t.sleep(0.5 * attempts)
                    continue
                # Non-rate-limit errors: stop early
                break
        if not results and last_err:
            raise RuntimeError(last_err)
        return results


class SerpAPIProvider:
    def __init__(self, session: requests.Session, api_key: str, timeout: int, max_retries: int) -> None:
        self.session = session
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "num": max_results, "api_key": self.api_key}
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=resp)
                resp.raise_for_status()
                data = resp.json()
                org = data.get("organic_results", [])
                results = [
                    {
                        "title": it.get("title", ""),
                        "snippet": it.get("snippet", ""),
                        "url": it.get("link", ""),
                    }
                    for it in org
                ]
                return results[:max_results]
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
                last_error = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                import time as _t
                _t.sleep(0.5 * attempt)
        raise RuntimeError(f"SerpAPI failed after {self.max_retries} attempts: {last_error}")


class TavilyProvider:
    """Tavily Search API provider (POST JSON).

    See https://docs.tavily.com/ for API details.
    """

    def __init__(self, session: requests.Session, api_key: str, timeout: int, max_retries: int) -> None:
        self.session = session
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max(1, int(max_results)),
            "search_depth": "basic",
        }
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=resp)
                resp.raise_for_status()
                data = resp.json() or {}
                items = data.get("results", []) or []
                results = [
                    {
                        "title": it.get("title", ""),
                        "snippet": it.get("content", ""),
                        "url": it.get("url", ""),
                    }
                    for it in items
                ]
                return results[:max_results]
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
                last_error = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                import time as _t
                _t.sleep(0.5 * attempt)
        raise RuntimeError(f"Tavily failed after {self.max_retries} attempts: {last_error}")


class BingSearchProvider:
    """Bing Search API provider for web search fallback."""
    
    def __init__(self, session: requests.Session, api_key: str, timeout: int, max_retries: int) -> None:
        self.session = session
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": max_results, "responseFilter": "webpages"}
        
        attempt = 0
        last_error: Optional[Exception] = None
        
        while attempt < self.max_retries:
            try:
                resp = self.session.get(
                    self.base_url, 
                    params=params, 
                    headers=headers, 
                    timeout=self.timeout
                )
                
                if resp.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=resp)
                
                resp.raise_for_status()
                data = resp.json()
                
                web_pages = data.get("webPages", {}).get("value", [])
                results = [
                    {
                        "title": page.get("name", ""),
                        "snippet": page.get("snippet", ""),
                        "url": page.get("url", ""),
                    }
                    for page in web_pages
                ]
                return results[:max_results]
                
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
                last_error = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                import time as _t
                _t.sleep(0.5 * attempt)
                
        raise RuntimeError(f"Bing Search failed after {self.max_retries} attempts: {last_error}")


class Calculator(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )

    def execute(self, expression: str) -> Dict:
        try:
            import ast

            allowed = {
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }

            def _eval(node):
                if isinstance(node, ast.Expression):
                    return _eval(node.body)
                if isinstance(node, (ast.Num, getattr(ast, 'Constant', ast.Num))):
                    return getattr(node, 'n', getattr(node, 'value', None))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    val = _eval(node.operand)
                    return +val if isinstance(node.op, ast.UAdd) else -val
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                    left, right = _eval(node.left), _eval(node.right)
                    return {
                        ast.Add: left + right,
                        ast.Sub: left - right,
                        ast.Mult: left * right,
                        ast.Div: left / right,
                        ast.Pow: left ** right,
                        ast.Mod: left % right,
                    }[type(node.op)]
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func = allowed.get(node.func.id)
                    if not func:
                        raise ValueError("Unsupported function")
                    args = [_eval(a) for a in node.args]
                    return func(*args)
                if isinstance(node, ast.Name):
                    if node.id in allowed:
                        return allowed[node.id]
                    raise ValueError("Unknown identifier")
                raise ValueError("Unsupported expression")

            parsed = ast.parse(expression, mode="eval")
            result = _eval(parsed)

            return {
                "expression": expression,
                "result": result,
                "formatted_result": f"{result:,}" if isinstance(result, (int, float)) else str(result),
            }
        except Exception as e:
            return {"expression": expression, "error": f"Invalid expression: {str(e)}"}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression using +,-,*,/,**,mod and sin,cos,tan,sqrt,log,log10,exp,pi,e"}
            },
            "required": ["expression"],
        }


class TimezoneTool(Tool):
    def __init__(self):
        super().__init__(
            name="timezone",
            description="Convert between timezones"
        )

    def execute(self, time: str, from_tz: str, to_tz: str) -> str:
        """
        Convert time between different timezones.

        Args:
            time (str): ISO-8601 formatted datetime string.
            from_tz (str): Source timezone.
            to_tz (str): Target timezone.
        Returns:
            str: JSON string containing original and converted time info,
                 or an error message if conversion fails.
        """
        try:
            # Parse the ISO-8601 time string, replacing 'Z' with '+00:00'
            dt = datetime.fromisoformat(time.replace('Z', '+00:00'))
            
            # Localize the datetime to the source timezone without any tz info
            source_timezone = pytz.timezone(from_tz)
            dt_source = source_timezone.localize(dt.replace(tzinfo=None))
            
            # Convert the localized time to the target timezone
            target_timezone = pytz.timezone(to_tz)
            dt_target = dt_source.astimezone(target_timezone)
            
            result = {
                "original_time": time,
                "original_timezone": from_tz,
                "converted_time": dt_target.isoformat(),
                "converted_timezone": to_tz
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error converting timezone: {str(e)}"
    
    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "ISO-8601 datetime"},
                "from_tz": {"type": "string", "description": "Source timezone"},
                "to_tz": {"type": "string", "description": "Target timezone"},
            },
            "required": ["time", "from_tz", "to_tz"],
        }

class UnitConverter(Tool):
    def __init__(self):
        super().__init__(
            name="unit_converter",
            description="Convert between different units"
        )

    def execute(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert values between different units of measurement."""
        # Simple mapping of conversion factors for common units
        conversion_map = {
            # Length
            "m_to_ft": 3.28084,
            "ft_to_m": 0.3048,
            "km_to_mile": 0.621371,
            "mile_to_km": 1.60934,
            # Weight/Mass
            "kg_to_lb": 2.20462,
            "lb_to_kg": 0.453592,
            # Temperature needs special handling
            "celsius_to_fahrenheit": lambda c: c * 9/5 + 32,
            "fahrenheit_to_celsius": lambda f: (f - 32) * 5/9,
        }
        
        try:
            # Create a key for the conversion map
            conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"
            
            # Check if conversion exists
            if conversion_key in conversion_map:
                conversion = conversion_map[conversion_key]
                
                # Handle functions (e.g., temperature conversions)
                if callable(conversion):
                    result = conversion(value)
                else:
                    result = value * conversion
                    
                return {
                    "original_value": value,
                    "original_unit": from_unit,
                    "converted_value": result,
                    "converted_unit": to_unit
                }
            else:
                return f"Conversion from {from_unit} to {to_unit} is not supported"
                
        except Exception as e:
            return f"Error converting units: {str(e)}"
        
    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        }

class WebScraper(NetworkTool):
    """
    Tool to scrape data from a webpage. This tool fetches the HTML content of a given URL and returns the text content.
    """
    def __init__(self, timeout: int = 15, max_retries: int = 3, security_manager: Optional[SecurityManager] = None):
        super().__init__(
            name="web_scraper",
            description="Scrape data from a webpage and return the text content",
            timeout=timeout,
            max_retries=max_retries,
            security_manager=security_manager,
        )

    def execute(self, url: str) -> Dict:
        """
        Scrape data from a webpage.
        """
        try:
            if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
                return {"error": "Invalid URL. Must start with http:// or https://"}
            resp = self.make_request("GET", url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text()
            # Clean up the text content
            text = ' '.join(text.split())
            text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Return the text content (trimmed)
            return {
                "url": url,
                "content": text[:1000]
            }
        except NetworkError as e:
            return {"error": f"Network error: {e}"}
        except Exception as e:
            return {"error": f"Scrape failed: {e}"}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch and extract text from"}
            },
            "required": ["url"],
        }

class DateTimeTool(Tool):
    def __init__(self):
        super().__init__(name="datetime", description="Get current time, add offsets, or format")

    def execute(self, action: str = "now", offset_seconds: int = 0, fmt: str = None) -> Dict:
        try:
            now = datetime.utcnow()
            if action == "add":
                now = now + __import__("datetime").timedelta(seconds=offset_seconds)
            if fmt:
                # Convert common format patterns to Python strftime format
                python_fmt = self._convert_format(fmt)
                formatted_date = now.strftime(python_fmt)
                return {"result": formatted_date, "iso": now.isoformat() + "Z"}
            return {"iso": now.isoformat() + "Z"}
        except Exception as e:
            return {"error": str(e)}
    
    def _convert_format(self, fmt: str) -> str:
        """Convert common date format patterns to Python strftime format."""
        # Order matters - replace longer patterns first
        conversions = [
            ('yyyy', '%Y'),
            ('MMMM', '%B'),  # Full month name (before MMM)
            ('MMM', '%b'),   # Abbreviated month name (before MM)
            ('MM', '%m'),    # Month number (after month names)
            ('dd', '%d'),
            ('HH', '%H'),
            ('mm', '%M'),
            ('ss', '%S'),
        ]
        
        result = fmt
        for pattern, replacement in conversions:
            result = result.replace(pattern, replacement)
        
        return result

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["now", "add"], "default": "now"},
                "offset_seconds": {"type": "integer", "default": 0},
                "fmt": {"type": "string", "description": "strftime format string"},
            },
        }

class UUIDTool(Tool):
    def __init__(self):
        super().__init__(name="uuid", description="Generate UUID v4 strings")

    def execute(self, count: int = 1) -> Dict:
        import uuid
        try:
            count = max(1, min(int(count), 20))
            uuids = [str(uuid.uuid4()) for _ in range(count)]
            return {"uuids": uuids}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"count": {"type": "integer", "default": 1}}}

class RandomStringTool(Tool):
    def __init__(self):
        super().__init__(name="random_string", description="Generate a random alphanumeric string")

    def execute(self, length: int = 16) -> Dict:
        import secrets
        import string
        try:
            length = max(1, min(int(length), 1024))
            alphabet = string.ascii_letters + string.digits
            s = "".join(secrets.choice(alphabet) for _ in range(length))
            return {"value": s}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"length": {"type": "integer", "default": 16}}}

class RandomNumberTool(Tool):
    def __init__(self):
        super().__init__(name="random_number", description="Generate a random integer in a range")

    def execute(self, minimum: int = 0, maximum: int = 100) -> Dict:
        import random
        try:
            a, b = int(minimum), int(maximum)
            if a > b:
                a, b = b, a
            return {"value": random.randint(a, b), "min": a, "max": b}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"minimum": {"type": "integer", "default": 0}, "maximum": {"type": "integer", "default": 100}}}

class Base64EncodeTool(Tool):
    def __init__(self):
        super().__init__(name="base64_encode", description="Base64-encode a string")

    def execute(self, text: str) -> Dict:
        import base64
        try:
            data = base64.b64encode(text.encode("utf-8")).decode("ascii")
            return {"encoded": data}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

class Base64DecodeTool(Tool):
    def __init__(self):
        super().__init__(name="base64_decode", description="Base64-decode a string")

    def execute(self, data: str) -> Dict:
        import base64
        try:
            text = base64.b64decode(data.encode("ascii")).decode("utf-8")
            return {"text": text}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]}

class HashTool(Tool):
    def __init__(self):
        super().__init__(name="hash_text", description="Hash text with a selected algorithm")

    def execute(self, text: str, algorithm: str = "sha256") -> Dict:
        import hashlib
        try:
            alg = algorithm.lower()
            if alg not in hashlib.algorithms_available:
                return {"error": f"Unsupported algorithm: {algorithm}"}
            h = hashlib.new(alg)
            h.update(text.encode("utf-8"))
            return {"algorithm": alg, "hexdigest": h.hexdigest()}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "algorithm": {"type": "string", "default": "sha256"},
            },
            "required": ["text"],
        }

class JSONTool(Tool):
    def __init__(self):
        super().__init__(name="json_tool", description="Pretty-print, minify or validate JSON")

    def execute(self, text: str, mode: str = "pretty") -> Dict:
        import json
        try:
            obj = json.loads(text)
            if mode == "pretty":
                return {"result": json.dumps(obj, indent=2, ensure_ascii=False)}
            elif mode == "minify":
                return {"result": json.dumps(obj, separators=(",", ":"), ensure_ascii=False)}
            else:
                return {"valid": True}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "mode": {"type": "string", "enum": ["pretty", "minify", "validate"], "default": "pretty"},
            },
            "required": ["text"],
        }

class RegexExtractTool(Tool):
    def __init__(self):
        super().__init__(name="regex_extract", description="Extract regex matches from text")

    def execute(self, text: str, pattern: str, flags: str = "") -> Dict:
        import re
        try:
            f = 0
            if "i" in flags.lower():
                f |= re.IGNORECASE
            if "m" in flags.lower():
                f |= re.MULTILINE
            if "s" in flags.lower():
                f |= re.DOTALL
            matches = re.findall(pattern, text, flags=f)
            return {"matches": matches}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "pattern": {"type": "string"},
                "flags": {"type": "string", "description": "Combine: i (ignorecase), m (multiline), s (dotall)"},
            },
            "required": ["text", "pattern"],
        }

class LoremIpsumTool(Tool):
    def __init__(self):
        super().__init__(name="lorem_ipsum", description="Generate placeholder Lorem Ipsum text")

    def execute(self, sentences: int = 3) -> Dict:
        try:
            sentences = max(1, min(int(sentences), 20))
            base = (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
            )
            out = " ".join([base for _ in range(sentences)])
            return {"text": out}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"sentences": {"type": "integer", "default": 3}}}

class FileReadTool(Tool):
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        super().__init__(name="file_read", description="Read a small text file from the workspace")
        self.security = security_manager

    def execute(self, path: str, max_bytes: int = 4096) -> Dict:
        import os
        try:
            max_bytes = max(1, min(int(max_bytes), 1024 * 1024))
            if self.security:
                abs_path = self.security.validate_file_access(path)
            else:
                abs_path = os.path.abspath(path)
                cwd = os.path.abspath(os.getcwd())
                if not abs_path.startswith(cwd):
                    return {"error": "Access denied: path outside workspace"}
            if not os.path.exists(abs_path):
                return {"error": "File not found"}
            with open(abs_path, "rb") as f:
                data = f.read(max_bytes)
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1", errors="replace")
            return {"path": abs_path, "preview": text}
        except ToolExecutionError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 4096},
            },
            "required": ["path"],
        }

class FileSearchTool(Tool):
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        super().__init__(name="file_search", description="Search for files by pattern within the workspace")
        self.security = security_manager

    def execute(self, pattern: str = "*.md", max_results: int = 50) -> Dict:
        import os
        import fnmatch
        try:
            max_results = max(1, min(int(max_results), 1000))
            roots: List[str]
            if self.security and self.security.allowed_paths:
                roots = sorted(self.security.allowed_paths)
            else:
                roots = [os.path.abspath(os.getcwd())]
            matches: List[str] = []
            for root in roots:
                for curr_root, _dirs, files in os.walk(root):
                    for name in files:
                        if fnmatch.fnmatch(name, pattern):
                            matches.append(os.path.join(curr_root, name))
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break
            return {"pattern": pattern, "results": matches}
        except ToolExecutionError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"pattern": {"type": "string", "default": "*.md"}, "max_results": {"type": "integer", "default": 50}}}

class CSVPreviewTool(Tool):
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        super().__init__(name="csv_preview", description="Preview CSV headers and first rows")
        self.security = security_manager

    def execute(self, path: str, limit: int = 5) -> Dict:
        import os
        import csv
        try:
            if self.security:
                abs_path = self.security.validate_file_access(path)
            else:
                abs_path = os.path.abspath(path)
                cwd = os.path.abspath(os.getcwd())
                if not abs_path.startswith(cwd):
                    return {"error": "Access denied: path outside workspace"}
            if not os.path.exists(abs_path):
                return {"error": "File not found"}
            limit = max(1, min(int(limit), 100))
            with open(abs_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            headers = rows[0] if rows else []
            preview = rows[1: 1 + limit] if len(rows) > 1 else []
            return {"path": abs_path, "headers": headers, "rows": preview}
        except ToolExecutionError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
            "required": ["path"],
        }

class URLMetadataTool(NetworkTool):
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        super().__init__(
            name="url_metadata",
            description="Fetch title and meta description for a URL",
            timeout=timeout,
            max_retries=max_retries,
        )

    def execute(self, url: str) -> Dict:
        try:
            if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
                return {"error": "Invalid URL. Must start with http:// or https://"}
            resp = self.make_request("GET", url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else None
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else None
            return {"url": url, "title": title, "description": desc}
        except NetworkError as e:
            return {"error": f"Network error: {e}"}
        except Exception as e:
            return {"error": f"Metadata fetch failed: {e}"}

    def _get_parameters(self) -> Dict:
        return {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}


class ToolsHelpTool(Tool):
    def __init__(self, tool_manager):
        super().__init__(name="tools_help", description="List available tools and their parameter schemas")
        self._tool_manager = tool_manager

    def execute(self, names: Optional[List[str]] = None, include_schema: bool = True) -> Dict:
        try:
            defs = self._tool_manager.get_tool_definitions()
            if names:
                names_set = set(names)
                defs = [d for d in defs if d.get("function", {}).get("name") in names_set]
            if include_schema:
                return {"tools": defs}
            # Strip parameters if not requested
            simple = []
            for d in defs:
                fn = d.get("function", {})
                simple.append({"name": fn.get("name"), "description": fn.get("description")})
            return {"tools": simple}
        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}},
                "include_schema": {"type": "boolean", "default": True},
            },
        }

class WebHeadlinesTool(NetworkTool):
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        super().__init__(
            name="web_headlines",
            description="Extract H1/H2/H3 headings from a webpage",
            timeout=timeout,
            max_retries=max_retries,
        )

    def execute(self, url: str, max_items: int = 10) -> Dict:
        try:
            if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
                return {"error": "Invalid URL. Must start with http:// or https://"}
            resp = self.make_request("GET", url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            heads = []
            for tag in soup.find_all(["h1", "h2", "h3"]):
                text = (tag.get_text() or "").strip()
                if text:
                    heads.append({"tag": tag.name, "text": text})
                if len(heads) >= max_items:
                    break
            return {"url": url, "headings": heads}
        except NetworkError as e:
            return {"error": f"Network error: {e}"}
        except Exception as e:
            return {"error": f"Headlines fetch failed: {e}"}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_items": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        }
