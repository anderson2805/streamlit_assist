# Assist Streamlit Application

A standalone Streamlit application that provides a user-friendly interface for exploring prompt templates and conducting web searches based on the Government Communication Service (GCS) themes and use cases.

## Features

- **Theme Exploration**: Browse through different communication areas like "Planning campaigns", "Research", "Media handling", etc.
- **Use Case Selection**: Choose specific use cases within each theme
- **Prompt Templates**: Structured prompts based on the selected use case
- **Web Search Integration**: Optional web search functionality to enhance responses
- **Chat Interface**: Interactive chat interface to input data and receive responses
- **Session Management**: Maintains chat history within the session

## File Structure

```
streamlit_assist/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Data files (copied from assist_service)
    ├── themes/
    │   └── 2024-10-10_themes.json
    ├── use_cases/
    │   └── 2024-10-10_use_cases.json
    └── prompts/
        └── 2024-10-10_prompts.json
```

## Installation

1. Navigate to the streamlit_assist directory:
   ```bash
   cd streamlit_assist
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the data files are present in the `data/` directory. If not, copy them from the main assist_service:
   ```bash
   # From the main project directory
   copy assist_service\data\themes\2024-10-10_themes.json streamlit_assist\data\themes\
   copy assist_service\data\use_cases\2024-10-10_use_cases.json streamlit_assist\data\use_cases\
   copy assist_service\data\prompts\2024-10-10_prompts.json streamlit_assist\data\prompts\
   ```

## Running the Application

1. Navigate to the streamlit_assist directory:
   ```bash
   cd streamlit_assist
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`)

## Usage

### 1. Explore Areas Tab
- Browse through different theme areas (e.g., "Planning campaigns and communication", "Research", etc.)
- Click on a theme to see its available use cases
- Select a use case to proceed to the chat interface

### 2. Chat Tab
- View the selected use case and its instructions
- Enable web search if needed (currently a placeholder)
- Enter your input based on the provided form template
- Generate responses using the structured prompt templates
- View chat history of previous interactions

### 3. Additional Tabs
- Upload Documents: Placeholder for document upload functionality
- Manage Prompts: Placeholder for prompt management

## Technical Implementation

### Key Components

1. **Data Loading**: Uses `@st.cache_data` to efficiently load JSON data files
2. **Session State**: Maintains user selections and chat history across page interactions
3. **Responsive UI**: Custom CSS for modern, professional appearance
4. **Template System**: Dynamic prompt template filling based on user input

### Web Search Integration

The application now includes **enhanced web search functionality** based on the `chat_enhance_user_prompt.py` component:

#### Features:
- **Enhanced Mode**: Simulates GOV.UK search API + web browsing functionality
- **Real-time Mode**: Uses DuckDuckGo Instant Answer API for live web results
- **URL Extraction**: Automatically extracts and processes URLs from user queries
- **Search Term Analysis**: Intelligently extracts relevant search terms
- **Citations Management**: Tracks and displays source citations
- **Error Handling**: Graceful fallback when search APIs are unavailable

#### Implementation:
- `WebSearchService`: Simulates the sophisticated logic from `chat_enhance_user_prompt.py`
- `RealTimeWebSearch`: Provides actual web search using public APIs
- Automatic content enhancement and citation tracking

### AI Response Generation

The current implementation shows how prompts are structured and filled. To add actual AI responses:

1. Integrate with an LLM API (e.g., OpenAI, Anthropic, Azure OpenAI)
2. Replace the `generate_response()` function with actual API calls
3. Add API key configuration and error handling

## Customization

### Styling
- Modify the CSS in the `st.markdown()` section at the top of `app.py`
- Colors, fonts, and layout can be customized

### Data Sources
- Update the JSON file paths in the load functions if using different data sources
- The application automatically adapts to the structure of the JSON files

### Additional Features
- Add new tabs by modifying the `st.tabs()` section
- Extend functionality by adding new functions and UI components

## Dependencies

- `streamlit>=1.28.0`: Main web application framework
- `requests>=2.31.0`: For HTTP requests (web search functionality)
- `python-dotenv>=1.0.0`: For environment variable management

## Notes

- This is a standalone application separate from the main assist_service
- The current implementation focuses on prompt template demonstration
- Web search and AI response generation are implemented as placeholders
- The UI closely follows the design patterns from the reference React application 