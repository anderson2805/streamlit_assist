# Assist Streamlit Application

A Streamlit application for exploring prompt templates and generating AI-powered responses with web search integration based on Government Communication Service (GCS) themes and use cases.

## Features

- **Theme & Use Case Selection**: Browse communication areas and select specific use cases
- **AI Response Generation**: OpenAI integration for intelligent responses
- **Web Search Integration**: Perplexity API with multiple search modes (Enhanced, Government-focused, Combined)
- **Chat Interface**: Interactive interface with session management and citation tracking
- **Streaming Responses**: Real-time response streaming

## File Structure

```
streamlit_assist/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── clients/              # AI and search service clients
│   ├── openai_client.py   # OpenAI API integration
│   └── perplexity_client.py # Perplexity search API integration
└── data/                 # JSON data files
    ├── themes/2024-10-10_themes.json
    ├── use_cases/2024-10-10_use_cases.json
    └── prompts/2024-10-10_prompts.json
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Explore Areas**: Browse themes and select use cases
2. **Chat Interface**: Configure AI settings, enter input, and generate responses
3. **Web Search**: Choose search modes for enhanced context
4. **Citations**: View sources from web search results

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI response generation  
- `PERPLEXITY_API_KEY`: Required for web search functionality

Both API keys are optional - the application handles missing keys gracefully.

## Dependencies

- `streamlit>=1.28.0`: Web application framework
- `requests>=2.31.0`: HTTP requests
- `python-dotenv>=1.0.0`: Environment variables 