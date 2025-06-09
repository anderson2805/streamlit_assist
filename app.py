import streamlit as st
import json
import os
import requests
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from clients.perplexity_client import PerplexityClient
from clients.openai_client import OpenAIClient

# Page configuration
st.set_page_config(
    page_title="Assist - Initiative",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #8e44ad, #6c5ce7);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .theme-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #8e44ad;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .theme-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .theme-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .theme-subtitle {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    .use-case-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #e9ecef;
    }
    
    .beta-badge {
        background: #3498db;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 1rem;
    }
    
    .status-badge {
        background: #95a5a6;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_themes():
    """Load themes from JSON file"""
    try:
        with open('data/themes/2024-10-10_themes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def load_use_cases():
    """Load use cases from JSON file"""
    try:
        with open('data/use_cases/2024-10-10_use_cases.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def load_prompts():
    """Load prompts from JSON file"""
    try:
        with open('data/prompts/2024-10-10_prompts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

async def generate_response(prompt: str, user_input: str, use_web_search: bool = False, search_type: str = "enhanced", stream: bool = False) -> Dict[str, Any]:
    """Generate response using the prompt template with OpenAI client"""
    # Append user input to the instruction prompt
    filled_prompt = f"{prompt}\n\nUser Input:\n{user_input}"
    
    try:
        # Initialize OpenAI client
        try:
            openai_client = OpenAIClient()
        except ValueError as e:
            return {
                "response": f"OpenAI client configuration error: {str(e)}. Please check your API key configuration.",
                "citations": [],
                "web_search_used": use_web_search
            }
        
        # Prepare messages for the chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant that follows the given instructions carefully."},
            {"role": "user", "content": filled_prompt}
        ]
        
        citations = []
        web_search_used = False
        
        if use_web_search:
            web_search_used = True
            try:
                # Initialize Perplexity client for all search types
                perplexity_client = PerplexityClient()
                
                if search_type == "both":
                    # Execute both searches and merge results
                    global_search_result = await perplexity_client.search_async(
                        query=user_input,
                        search_context_size="high",
                        search_domain_filter=None  # No domain filter for broader search
                    )
                    
                    gov_search_result = await perplexity_client.search_async(
                        query=user_input,
                        search_context_size="high",
                        search_domain_filter=["gov.sg"]
                    )
                    
                    # Merge search contexts
                    combined_context = ""
                    if global_search_result.get('content'):
                        combined_context += f"\n\nGlobal Web Search Context:\n{global_search_result['content']}"
                    if gov_search_result.get('content'):
                        combined_context += f"\n\nSingapore Government Context:\n{gov_search_result['content']}"
                    
                    if combined_context:
                        messages.append({"role": "user", "content": f"Additional context from multiple sources: {combined_context}"})
                    
                    # Merge citations from both searches
                    citations = []
                    citation_counter = 1
                    
                    # Add global search citations
                    if global_search_result.get('citations'):
                        for cite in global_search_result['citations']:
                            citations.append({
                                'docname': cite.get('title', f"Global Source {citation_counter}"),
                                'docurl': cite.get('url', ''),
                                'domain': cite.get('domain', ''),
                                'source_type': 'Global'
                            })
                            citation_counter += 1
                    
                    # Add government search citations
                    if gov_search_result.get('citations'):
                        for cite in gov_search_result['citations']:
                            citations.append({
                                'docname': cite.get('title', f"Gov.sg Source {citation_counter}"),
                                'docurl': cite.get('url', ''),
                                'domain': cite.get('domain', ''),
                                'source_type': 'Singapore Government'
                            })
                            citation_counter += 1
                
                elif search_type == "real_time":
                    # Use Perplexity search with no domain filter for global real-time search
                    search_result = await perplexity_client.search_async(
                        query=user_input,
                        search_context_size="high",
                        search_domain_filter=None  # No domain filter for broader search
                    )
                    
                    # Add web search context to messages
                    if search_result.get('content'):
                        web_context = f"\n\nGlobal Web Search Context:\n{search_result['content']}"
                        messages.append({"role": "user", "content": f"Additional context from global web search: {web_context}"})
                        
                        # Extract citations from Perplexity response
                        if search_result.get('citations'):
                            citations = [
                                {
                                    'docname': cite.get('title', f"Source {i}"),
                                    'docurl': cite.get('url', ''),
                                    'domain': cite.get('domain', ''),
                                    'source_type': 'Global'
                                }
                                for i, cite in enumerate(search_result['citations'], 1)
                            ]
                
                else:
                    # Use Perplexity search with gov.sg domain filter for Singapore government focused search
                    search_result = await perplexity_client.search_async(
                        query=user_input,
                        search_context_size="high",
                        search_domain_filter=["gov.sg"]
                    )
                    
                    # Add web search context to messages
                    if search_result.get('content'):
                        web_context = f"\n\nSingapore Government Context:\n{search_result['content']}"
                        messages.append({"role": "user", "content": f"Additional context from Singapore government sources: {web_context}"})
                        
                        # Extract citations from Perplexity response
                        if search_result.get('citations'):
                            citations = [
                                {
                                    'docname': cite.get('title', f"Source {i}"),
                                    'docurl': cite.get('url', ''),
                                    'domain': cite.get('domain', ''),
                                    'source_type': 'Singapore Government'
                                }
                                for i, cite in enumerate(search_result['citations'], 1)
                            ]
                        
            except Exception as e:
                # Add error context but continue with the request
                error_msg = f"Web search encountered an error: {str(e)}"
                messages.append({"role": "user", "content": f"Note: {error_msg}"})
        
        # Get chat completion from OpenAI
        if stream:
            # Use streaming method for real-time response
            response_text = ""
            async for chunk in openai_client.get_chat_completion_stream_async(
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            ):
                response_text += chunk
        else:
            # Use regular method (streaming is handled internally)
            response = await openai_client.get_chat_completion_async(
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                stream=True
            )
            response_text = response.choices[0].message.content
        
        return {
            "response": response_text,
            "citations": citations,
            "web_search_used": web_search_used
        }
        
    except Exception as e:
        return {
            "response": f"Sorry, I encountered an error while generating the response: {str(e)}",
            "citations": [],
            "web_search_used": use_web_search
        }

# Initialize session state
if 'selected_theme' not in st.session_state:
    st.session_state.selected_theme = None
if 'selected_use_case' not in st.session_state:
    st.session_state.selected_use_case = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Load data
themes = load_themes()
use_cases_data = load_use_cases()
prompts = load_prompts()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Assist</h1>
        <p>Initiative</p>
        <div style="margin-top: 1rem;">
            <span class="beta-badge">Beta</span>
            <span class="status-badge">Currently in development</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation using segmented control
    if 'active_section' not in st.session_state:
        st.session_state.active_section = "🏠 Explore Areas"
    
    # Navigation segmented control
    selected_section = st.segmented_control(
        "Navigation",
        options=["🏠 Explore Areas", "💬 Chat", "📄 Upload Documents", "⚙️ Manage Prompts"],
        default=st.session_state.active_section,
        label_visibility="collapsed"
    )
    
    # Update session state when selection changes
    if selected_section != st.session_state.active_section:
        st.session_state.active_section = selected_section
    
    # Show content based on selected section
    if selected_section == "🏠 Explore Areas":
        show_explore_areas()
    elif selected_section == "💬 Chat":
        show_chat_interface()
    elif selected_section == "📄 Upload Documents":
        st.info("Document upload functionality would be implemented here")
    elif selected_section == "⚙️ Manage Prompts":
        st.info("Prompt management functionality would be implemented here")

def show_explore_areas():
    """Show the main area exploration interface"""
    st.markdown("## Which area would you like to explore?")
    
    if not themes:
        st.error("No themes data found. Please check the data files.")
        return
    
    # Display themes in a grid
    cols = st.columns(2)
    
    for idx, theme in enumerate(themes):
        with cols[idx % 2]:
            if st.button(
                f"**{theme['title']}**\n\n{theme['subtitle']}", 
                key=f"theme_{theme['uuid']}",
                use_container_width=True
            ):
                st.session_state.selected_theme = theme
                st.rerun()
    
    # Show use cases for selected theme
    if st.session_state.selected_theme:
        show_use_cases(st.session_state.selected_theme)

def show_use_cases(selected_theme: Dict[str, Any]):
    """Show use cases for the selected theme"""
    st.markdown(f"## Use Cases for: {selected_theme['title']}")
    
    # Find use cases for this theme
    theme_use_cases = None
    for uc_group in use_cases_data:
        if uc_group['theme.uuid'] == selected_theme['uuid']:
            theme_use_cases = uc_group['use_cases']
            break
    
    if not theme_use_cases:
        st.warning("No use cases found for this theme.")
        return
    
    for use_case in theme_use_cases:
        with st.expander(f"🎯 {use_case['use_case.title']}"):
            st.markdown(f"**Instructions:** {use_case['use_case.instruction']}")
            
            if use_case.get('use_case.user_input_form'):
                st.markdown("**Input Form:**")
                st.code(use_case['use_case.user_input_form'])
            
            if st.button(f"Select this use case", key=f"select_{use_case['use_case.uuid']}"):
                st.session_state.selected_use_case = use_case
                st.session_state.active_section = "💬 Chat"
                st.rerun()

def show_chat_interface():
    """Show the chat interface using Streamlit chat elements"""
    st.markdown("## 💬 Chat Interface")
    
    if not st.session_state.selected_use_case:
        st.info("Please select a use case from the 'Explore Areas' tab first.")
        return
    
    use_case = st.session_state.selected_use_case
    
    # Show current use case above chat interface
    st.markdown("### 🎯 Current Use Case")
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{use_case['use_case.title']}**")
            st.markdown(f"*{use_case['use_case.instruction']}*")
        with col2:
            if st.button("📝 Change Use Case", use_container_width=True):
                st.session_state.selected_use_case = None
                st.session_state.active_section = "🏠 Explore Areas"
                st.rerun()
    
    st.markdown("---")
    
    # Show input guidance
    st.markdown("**💡 How it works:** Your input will be appended to the use case instruction above when processing your request.")
    
    # Get pre-fill text from use case input form
    prefill_text = use_case.get('use_case.user_input_form', '')
    
    # Initialize session state for input text if not exists
    if 'input_text' not in st.session_state:
        st.session_state.input_text = prefill_text
    
    # Update prefill text when use case changes
    if st.session_state.get('last_use_case_uuid') != use_case.get('use_case.uuid'):
        st.session_state.input_text = prefill_text
        st.session_state.last_use_case_uuid = use_case.get('use_case.uuid')
    
    # Input area with pre-filled text and send button
    col1, col2 = st.columns([4, 1])
    with col1:
        # Disable text area if there's already chat history
        is_locked = len(st.session_state.chat_history) > 0
        prompt = st.text_area(
            "Enter your input:",
            value=st.session_state.input_text,
            height=100,
            placeholder="Enter your input to be processed with the selected use case...",
            key="chat_input_area",
            disabled=is_locked
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        if not is_locked:
            send_button = st.button("📤 Send", use_container_width=True, type="primary")
        else:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True, type="secondary")
            send_button = False  # Ensure send_button is False when locked
            # Search Options (moved from sidebar)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="display: flex; align-items: center; height: 38px;"><strong>Search Options:</strong></div>', unsafe_allow_html=True)
    with col2:
        gov_sg_search = st.checkbox("Gov SG Search", value=True, help="Search within Singapore government domains (gov.sg)")
    with col3:
        global_search = st.checkbox("Global Search", value=False, help="Search across all web sources globally")

    
    # Determine search type and if web search is enabled
    use_web_search = gov_sg_search or global_search
    if gov_sg_search and global_search:
        search_type = "both"
    elif global_search:
        search_type = "real_time"
    else:
        search_type = "enhanced"
    
    # Create a container for chat messages
    chat_container = st.container(height=500)
    
    # Display chat history
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("👋 Welcome! Ask me anything related to your selected use case.")
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.markdown(chat['input'])
                    st.caption(f"🕒 {chat['timestamp'].strftime('%H:%M:%S')}")
                
                # Assistant message
                with st.chat_message("assistant"):
                    st.markdown(chat['response'])
                    
                    # Show web search info and citations
                    if chat.get('web_search_enabled'):
                        search_mode = chat.get('search_type', 'enhanced')
                        if search_mode == "both":
                            search_icon = "🌐🇸🇬"
                            search_text = "both global and Singapore government sources"
                        elif search_mode == "real_time":
                            search_icon = "🌐"
                            search_text = "global sources"
                        else:
                            search_icon = "🇸🇬"
                            search_text = "Singapore government sources"
                        
                        st.info(f"{search_icon} Web search was enabled using {search_text}")
                        
                        # Show citations if available
                        if chat.get('citations'):
                            with st.expander("📚 Citations & Sources"):
                                for j, citation in enumerate(chat['citations'], 1):
                                    source_type_badge = ""
                                    if citation.get('source_type'):
                                        if citation['source_type'] == 'Global':
                                            source_type_badge = " 🌐"
                                        elif citation['source_type'] == 'Singapore Government':
                                            source_type_badge = " 🇸🇬"
                                    
                                    if citation['docurl']:
                                        st.markdown(f"{source_type_badge} **[{j}]** [{citation['docurl']}]({citation['docurl']})\n\n")

    
    # Chat input for follow-up questions
    chat_input = st.chat_input("Ask a follow-up question...")
    
    # Handle clear chat button
    if 'clear_button' in locals() and clear_button:
        st.session_state.chat_history = []
        st.session_state.input_text = prefill_text
        st.rerun()
    
    # Process input when send button is clicked or chat input is used
    if (send_button and prompt.strip()) or chat_input:
            # Determine which input to use
            input_text = chat_input if chat_input else prompt
            
            # Add user message to chat history immediately for better UX
            user_message = {
                'timestamp': datetime.now(),
                'use_case': use_case['use_case.title'],
                'input': input_text,
                'response': "",
                'citations': [],
                'web_search_enabled': use_web_search,
                'search_type': search_type if use_web_search else None
            }
            
            # Show user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(input_text)
                    st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
            
            # Generate response
            with st.status("Generating response...", expanded=True) as status:
                st.write("🤖 Processing your request...")
                if use_web_search:
                    if search_type == "both":
                        st.write("🔍 Searching both global sources and Singapore government sources...")
                    elif search_type == "real_time":
                        st.write("🔍 Searching global sources...")
                    else:
                        st.write("🔍 Searching Singapore government sources...")
                st.write("✍️ Generating response...")
                
                try:
                    response_data = asyncio.run(generate_response(
                        use_case['use_case.instruction'], 
                        input_text, 
                        use_web_search,
                        search_type
                    ))
                    
                    # Update the user message with response data
                    user_message.update({
                        'response': response_data['response'],
                        'citations': response_data['citations'],
                        'web_search_enabled': response_data['web_search_used']
                    })
                    
                    status.update(label="✅ Response generated!", state="complete")
                    
                except Exception as e:
                    user_message['response'] = f"Sorry, I encountered an error: {str(e)}"
                    status.update(label="❌ Error occurred", state="error")
                
                # Add to chat history
                st.session_state.chat_history.append(user_message)
            
            # Clear the input text after sending
            st.session_state.input_text = prefill_text
            
            # Rerun to show the new message
            st.rerun()

if __name__ == "__main__":
    main() 