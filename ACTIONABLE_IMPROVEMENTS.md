# YAMLLM Actionable Improvements Checklist

**Purpose:** Track specific, actionable improvements identified in the comprehensive repository review.  
**Status:** Living document - update as items are completed  
**Priority Levels:** P0 (Critical), P1 (Important), P2 (Nice-to-have)

---

## Quick Wins (1-2 Days)

### Documentation Fixes

- [ ] **P0: Fix README installation instructions** (1 hour)
  - **Issue:** References non-existent `pip install yamllm-core`
  - **Fix:** Update to `pip install -e .` for development or actual PyPI package name
  - **Files:** `README.md`, `docs/installation.md`, `docs/_site/examples.html`
  - **Lines:** README:8-9, docs/_site/examples.html:85-89

- [ ] **P0: Document all available tools** (2 hours)
  - **Issue:** README lists only 4 tools, but 22 exist
  - **Fix:** 
    - Document all 22 tool classes from `yamllm/tools/utility_tools.py`:
      - WeatherTool, WebSearch, Calculator, TimezoneTool, UnitConverter
      - WebScraper, DateTimeTool, UUIDTool, RandomStringTool, RandomNumberTool
      - Base64EncodeTool, Base64DecodeTool, HashTool, JSONTool, RegexExtractTool
      - LoremIpsumTool, FileReadTool, FileSearchTool, CSVPreviewTool
      - URLMetadataTool, ToolsHelpTool, WebHeadlinesTool
    - Create comprehensive tool listing with descriptions and examples
    - Update README.md sections 120-128
  - **Files:** `README.md`, `docs/tools.md`

- [ ] **P1: Add CLI debugging documentation** (2 hours)
  - **Issue:** No documentation for debugging flags or logging
  - **Fix:** Document logging levels, debug flags, tool output handling
  - **Files:** Create `docs/cli-reference.md`

- [ ] **P1: Create manifesto alignment status** (2 hours)
  - **Issue:** Unclear which manifesto promises are implemented
  - **Fix:** Create matrix showing implemented vs pending features
  - **Files:** Create `MANIFESTO_STATUS.md`

### Configuration & Examples

- [ ] **P1: Add minimal example configs** (2 hours)
  - **Issue:** No simple working examples in repo
  - **Fix:** Add configs for OpenAI, Anthropic, Google in `.config_examples/`
  - **Files:** `.config_examples/openai_minimal.yaml`, etc.

- [ ] **P1: Update example scripts** (2 hours)
  - **Issue:** Examples don't show event callbacks or tool planning
  - **Fix:** Add event_callback wiring to show thinking and tool execution
  - **Files:** `examples/full_cli_chat.py`, `examples/openai_example.py`

---

## Critical Bugs & Fixes (Week 1-2)

### Code Correctness

- [ ] **P0: Verify streaming tool method names** (1 hour)
  - **Issue:** Previous reviews noted method name mismatch
  - **Check:** Ensure `process_streaming_tool_calls` is called correctly
  - **Files:** `yamllm/core/llm.py` (search for streaming tool calls)

- [ ] **P0: Audit API key masking** (2 hours)
  - **Issue:** Keys might leak in error messages or console output
  - **Fix:** Search for API key usage and ensure masking with `mask_string()`
  - **Files:** All provider files, `yamllm/core/utils.py`

- [ ] **P0: Verify path traversal protections** (1 hour)
  - **Issue:** File tools need proper path sanitization
  - **Check:** Confirm `expanduser` + `realpath` + null-byte checks
  - **Files:** `yamllm/tools/utility_tools.py` (file operations)

- [ ] **P1: Test MCP async implementation** (4 hours)
  - **Issue:** Reviews noted async misuse in MCP client
  - **Check:** Verify all async methods are awaited properly
  - **Files:** `yamllm/mcp/client.py`, `yamllm/mcp/connector.py`
  - **Tests:** `tests/test_mcp_async.py`

### Error Handling

- [ ] **P1: Replace print with logging** (2 hours)
  - **Issue:** Library code uses `print()` instead of `logging`
  - **Fix:** Find all `print()` calls and replace with `logger.info/debug()`
  - **Files:** `yamllm/core/parser.py`, `yamllm/core/config_validator.py`
  - **Command:** `grep -r "print(" yamllm/ --include="*.py"`

- [ ] **P1: Standardize exception types** (1 day)
  - **Issue:** Inconsistent error handling across providers
  - **Fix:** Create unified exception hierarchy
  - **Files:** `yamllm/core/exceptions.py`
  - **Example:**
    ```python
    class YAMLLMError(Exception): pass
    class ProviderError(YAMLLMError): pass
    class ToolError(YAMLLMError): pass
    ```

---

## Architecture Refactoring (Weeks 3-6)

### LLM Class Decomposition

- [ ] **P0: Split monolithic LLM class** (2-3 weeks)
  - **Issue:** `yamllm/core/llm.py` is 1,548 lines
  - **Plan:**
    1. Extract `ResponseOrchestrator` (lines 482-529)
    2. Extract `StreamingManager` (streaming logic)
    3. Extract `ToolSelector` (lines 616-752)
    4. Reduce main class to <500 lines
  - **Files:** 
    - Create: `yamllm/core/response_orchestrator.py`
    - Create: `yamllm/core/streaming_manager.py`
    - Create: `yamllm/core/tool_selector.py`
    - Refactor: `yamllm/core/llm.py`
  - **Tests:** Update all tests in `tests/test_llm.py`

- [ ] **P0: Unify sync/async architecture** (2 weeks)
  - **Issue:** Duplication between `llm.py` (1,547 lines) and `async_llm.py` (270 lines)
  - **Plan:**
    1. Make LLM async-first
    2. Add sync wrapper methods (`query_sync()`)
    3. Merge `BaseProvider` and `AsyncBaseProvider`
    4. Remove `async_llm.py`
  - **Files:**
    - Refactor: `yamllm/core/llm.py`
    - Remove: `yamllm/core/async_llm.py`
    - Refactor: `yamllm/providers/base.py`
    - Remove: `yamllm/providers/async_base.py`
  - **Tests:** Update all provider tests

### Tool Management

- [ ] **P1: Consolidate dual ToolManager** (1 week)
  - **Issue:** Two classes named `ToolManager` with different purposes
  - **Plan:**
    1. Rename `yamllm/tools/manager.py:ToolManager` → `ToolExecutor`
    2. Keep `yamllm/core/tool_management.py:ToolRegistryManager`
    3. Document clear separation of concerns
  - **Files:**
    - Rename class in: `yamllm/tools/manager.py`
    - Update imports throughout codebase
  - **Tests:** Update `tests/test_tool_manager.py`

### CLI Modularization

- [ ] **P1: Break down monolithic CLI** (1 week)
  - **Issue:** `yamllm/cli.py` is 800+ lines
  - **Plan:**
    1. Create `yamllm/cli/` directory
    2. Split into: `tools.py`, `config.py`, `chat.py`, `memory.py`
    3. Create `main.py` for argparse assembly
  - **Files:**
    - Create: `yamllm/cli/tools.py`
    - Create: `yamllm/cli/config.py`
    - Create: `yamllm/cli/chat.py`
    - Create: `yamllm/cli/memory.py`
    - Create: `yamllm/cli/main.py`
    - Refactor: `yamllm/cli.py` → use new structure

---

## Testing Improvements (Weeks 3-8)

### Missing Critical Tests

- [ ] **P0: Add tool filtering tests** (1 day)
  - **Issue:** No tests for `_filter_tools_for_prompt()` (136-line function!)
  - **Create:** `tests/test_tool_filtering.py`
  - **Tests needed:**
    ```python
    def test_explicit_tool_extraction()
    def test_tool_filtering_with_context()
    def test_tool_blacklist_filtering()
    def test_tool_filtering_edge_cases()
    ```

- [ ] **P0: Add tool choice tests** (1 day)
  - **Issue:** No tests for `_determine_tool_choice()`
  - **Add to:** `tests/test_tool_filtering.py`
  - **Tests needed:**
    ```python
    def test_determine_tool_choice_required()
    def test_determine_tool_choice_auto()
    def test_determine_tool_choice_none()
    ```

- [ ] **P1: Add CLI integration tests** (2 days)
  - **Issue:** No end-to-end CLI tests
  - **Create:** `tests/integration/test_cli_flows.py`
  - **Tests needed:**
    ```python
    def test_cli_chat_with_tools()
    def test_cli_tool_visibility()
    def test_cli_error_handling()
    def test_cli_config_validation()
    ```

- [ ] **P1: Add MCP edge case tests** (1 day)
  - **Issue:** Basic MCP test exists but no edge cases
  - **Add to:** `tests/test_mcp_async.py`
  - **Tests needed:**
    ```python
    def test_mcp_connector_timeout()
    def test_mcp_malformed_response()
    def test_mcp_connection_failure()
    def test_mcp_tool_registration_failure()
    ```

- [ ] **P1: Add provider fallback tests** (1 day)
  - **Issue:** No tests for provider failure scenarios
  - **Create:** `tests/test_provider_fallback.py`
  - **Tests needed:**
    ```python
    def test_provider_failure_cascade()
    def test_partial_tool_failure_recovery()
    def test_provider_timeout_handling()
    ```

- [ ] **P1: Add memory concurrency tests** (1 day)
  - **Issue:** No thread safety tests for memory operations
  - **Add to:** `tests/test_thread_safety.py`
  - **Tests needed:**
    ```python
    def test_concurrent_memory_writes()
    def test_concurrent_vector_store_access()
    def test_memory_cleanup_edge_cases()
    ```

### Performance Testing

- [ ] **P1: Create performance test harness** (1 week)
  - **Issue:** No performance measurements or benchmarks
  - **Create:** `tests/performance/test_latency.py`
  - **Tests needed:**
    ```python
    def test_first_token_latency()  # Target: <350ms
    def test_throughput_requests_per_minute()
    def test_memory_usage_baseline()
    def test_embedding_cache_performance()
    ```

- [ ] **P2: Add UI snapshot tests** (1 week)
  - **Issue:** No tests for Rich UI rendering
  - **Create:** `tests/test_ui_snapshots.py`
  - **Requires:** pytest-regtest or similar

### Security Testing

- [ ] **P2: Expand security test coverage** (3 days)
  - **Issue:** Only basic path traversal tests
  - **Add to:** `tests/test_security.py`
  - **Tests needed:**
    ```python
    def test_api_key_masking_in_logs()
    def test_api_key_masking_in_errors()
    def test_null_byte_in_paths()
    def test_internal_ip_blocking()
    def test_local_domain_blocking()
    ```

---

## Performance Optimization (Weeks 5-8)

### Caching Improvements

- [ ] **P0: Increase embedding cache size** (1 hour)
  - **Issue:** Cache limited to 64 entries
  - **Fix:** Increase to 1000+ with TTL-based eviction
  - **File:** `yamllm/core/llm.py:1342-1343`
  - **Change:**
    ```python
    # Before:
    self.embedding_cache = {}  # Limited to 64
    
    # After:
    from functools import lru_cache
    @lru_cache(maxsize=1000)
    def _get_embedding_cached(self, text: str):
        # Add TTL if needed
    ```

- [ ] **P1: Cache tool definitions** (2 hours)
  - **Issue:** Tool schemas regenerated on every request
  - **Fix:** Cache based on config hash
  - **File:** `yamllm/core/tool_orchestrator.py`
  - **Change:**
    ```python
    from functools import lru_cache
    
    @lru_cache(maxsize=32)
    def _get_cached_tool_definitions(self, config_hash: str):
        return self.generate_tool_definitions()
    ```

### Connection Management

- [ ] **P1: Add HTTP connection pooling** (1 week)
  - **Issue:** Clients recreated for each request
  - **Fix:** Use httpx with connection pools
  - **Files:** All provider implementations
  - **Example:**
    ```python
    # Create persistent client
    self.client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100)
    )
    ```

### Smart Search

- [ ] **P2: Add vector store relevance gating** (3 days)
  - **Issue:** FAISS search on every query regardless of relevance
  - **Fix:** Check if search is needed before executing
  - **File:** `yamllm/core/memory_manager.py`

---

## Security Enhancements (Weeks 6-8)

### Rate Limiting

- [ ] **P1: Implement rate limiting** (1 week)
  - **Issue:** No protection against abuse
  - **Create:** `yamllm/core/rate_limiter.py`
  - **Implementation:**
    ```python
    from collections import deque
    import time
    
    class RateLimiter:
        def __init__(self, max_requests_per_minute=60):
            self.requests = deque()
            self.max_requests = max_requests_per_minute
        
        def check_rate_limit(self):
            now = time.time()
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            if len(self.requests) >= self.max_requests:
                raise RateLimitExceeded()
            
            self.requests.append(now)
    ```

### Input Validation

- [ ] **P2: Add tool parameter validation** (1 week)
  - **Issue:** Limited validation on tool inputs
  - **File:** `yamllm/tools/base.py`
  - **Add:**
    - Length checks
    - Type validation
    - Malicious pattern detection

### Audit Logging

- [ ] **P2: Implement audit logging** (1 week)
  - **Issue:** No audit trail for tool usage
  - **Create:** `yamllm/core/audit_logger.py`
  - **Log:** Tool invocations, API calls, errors

---

## Dependency Management (Week 3)

### Cleanup

- [ ] **P1: Remove unused dependencies** (2 hours)
  - **Issue:** `matplotlib`, `seaborn`, `scikit-learn` in optional deps
  - **File:** `pyproject.toml`
  - **Action:** Confirm usage, remove or document

- [ ] **P2: Standardize provider signatures** (1 day)
  - **Issue:** `BaseProvider.__init__` doesn't match implementations
  - **File:** `yamllm/providers/base.py`
  - **Fix:** Align constructor parameters

---

## Documentation Expansion (Weeks 9-10)

### API Documentation

- [ ] **P1: Generate API reference** (1 week)
  - **Tool:** Sphinx or MkDocs
  - **Output:** `docs/api/`
  - **Coverage:** 100% of public APIs

### Tutorials

- [ ] **P1: Create beginner tutorial** (3 days)
  - **File:** `docs/tutorials/getting-started.md`
  - **Content:** Installation to first query

- [ ] **P1: Create advanced tutorial** (3 days)
  - **File:** `docs/tutorials/advanced-usage.md`
  - **Content:** Tools, memory, streaming

### Examples

- [ ] **P2: Add example for each provider** (1 week)
  - **Directory:** `examples/providers/`
  - **Files:** `openai.py`, `anthropic.py`, `google.py`, etc.

### FAQ

- [ ] **P2: Create FAQ document** (2 days)
  - **File:** `docs/FAQ.md`
  - **Content:** Top 20 common questions

---

## Monitoring & Observability (Weeks 11-12)

### Metrics

- [ ] **P2: Add performance metrics** (1 week)
  - **Create:** `yamllm/core/metrics.py`
  - **Export:** Prometheus format
  - **Metrics:**
    - Request latency
    - Token throughput
    - Tool execution time
    - Cache hit rates

### Logging

- [ ] **P2: Structured logging** (3 days)
  - **Issue:** Inconsistent log formats
  - **Fix:** Use JSON structured logs
  - **File:** `yamllm/core/logger.py`

---

## Release Preparation (Week 12)

### Pre-release Checklist

- [ ] **All P0 items completed**
- [ ] **Test coverage >80%**
- [ ] **Documentation complete**
- [ ] **Performance benchmarks passing**
- [ ] **Security audit complete**
- [ ] **CHANGELOG.md updated**
- [ ] **Version bumped to 1.0.0**

### CI/CD

- [ ] **P1: GitHub Actions workflows** (1 day)
  - **File:** `.github/workflows/ci.yml`
  - **Jobs:**
    - Lint (ruff, black, mypy)
    - Test (pytest)
    - Coverage (pytest-cov)
    - Build (package)

### Publishing

- [ ] **P1: Publish to PyPI** (1 day)
  - **Ensure:** Package name is available
  - **Update:** Installation instructions
  - **Test:** Installation from PyPI

---

## Progress Tracking

### Completed Items ✅
- (Track completed items here as work progresses)

### In Progress 🔄
- (Track items currently being worked on)

### Blocked ⛔
- (Track items waiting on dependencies)

---

## Notes

- **Priority levels:**
  - **P0:** Must be done, critical for production
  - **P1:** Should be done, important for quality
  - **P2:** Nice to have, improves experience

- **Time estimates:**
  - Based on single developer working full-time
  - Adjust for team size and availability
  - Some items can be parallelized

- **Dependencies:**
  - Architecture refactoring should complete before feature additions
  - Documentation can happen in parallel
  - Testing should be continuous

---

**Last Updated:** December 2024  
**Maintainers:** YAMLLM Project Team
