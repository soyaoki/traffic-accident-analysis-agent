# Project Conventions

## LLM Configuration
- **Default Model**: `gemini-2.5-flash` via AI Studio (Google).
- **Fallback/Alternative**: `gpt-4o-mini` (OpenAI).
- **OpenAI Compatibility**: The project uses the OpenAI-compatible interface for Gemini. When using Gemini, `base_url` must be set to the AI Studio or Vertex AI endpoint.

## Agent Architecture
- **Multi-Agent Flow**: `TrafficSafetyAnalyst` (Layer 2) coordinates and interprets, while `DataQueryAgent` (Layer 1) executes data operations via tools.
- **Streaming**: Implementation should prefer `Runner.run_streamed` to provide real-time feedback in the UI.

## Dashboard (Streamlit)
- **Visibility**: Always show intermediate agent thoughts and tool calls using `st.status`.
- **Transparency**: Provide an option (expander) to view raw JSON outputs from tools.
- **Robustness**: Disable agent tracing (`set_tracing_disabled(True)`) if `OPENAI_API_KEY` is not a valid `sk-` key to prevent startup errors.

## Data Layer
- **Tools**: Analysis tools are defined in `src/tools.py` using the `@function_tool` decorator.
- **Versatility**: Use `filter_accidents` for specific drill-downs and `aggregate_accidents` for SQL-like group-by operations.
- **Pre-processing**: Logic for loading and labeling CSVs is in `src/preprocess.py`.
