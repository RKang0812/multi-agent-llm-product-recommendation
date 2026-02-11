"""
Web Search Agent

Standalone agent for web search and product information gathering using OpenAI.
"""

import streamlit as st
from langchain_openai import ChatOpenAI
import os
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set Serper key
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY


def main():
    """Main function"""
    st.title("🔍 Web Search Agent")
    st.write("Search the web for product information, reviews, and comparisons")
    
    # Validate API key
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found! Please add it to your .env file.")
        return
    
    # Create LLM instance
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_API_KEY,
        max_tokens=2000
    )
    
    # Initialize search tool
    serper_tool = SerperDevTool()
    
    # Create search agent
    search_agent = Agent(
        role="Web Search Specialist",
        goal="Search and gather comprehensive information from the web",
        backstory="""You are an expert web search agent who knows how to find accurate,
        relevant information. You provide comprehensive answers with credible sources and links.
        When searching for products, you include purchase information, prices, and reputable sellers.""",
        tools=[serper_tool],
        verbose=True,
        llm=llm,
        allow_delegation=True
    )
    
    # User input
    st.write("### Enter Your Search Query")
    search_query = st.text_input(
        "What do you want to search for?",
        placeholder="Example: Best wireless headphones 2025"
    )
    
    if search_query:
        st.write(f"**Searching for:** {search_query}")
        
        # Create search task
        search_task = Task(
            description=f"""
            Search for comprehensive information about: "{search_query}"
            
            Provide:
            1. Detailed answer to the query
            2. Key findings and insights
            3. Credible sources and references
            4. If product-related: prices and purchase links
            5. Product comparisons if applicable
            6. User reviews and ratings
            7. Summary of main points
            
            Use web search to find current, accurate information.
            """,
            agent=search_agent,
            tools=[serper_tool],
            expected_output="Comprehensive search results with detailed information, sources, and links."
        )
        
        # Create crew
        crew = Crew(
            tasks=[search_task],
            agents=[search_agent],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute search
        with st.spinner("🔍 Searching the web..."):
            try:
                result = crew.kickoff()
                
                # Display results
                st.write("---")
                st.write("### ✅ Search Results")
                st.write(str(result))
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.info("Please check your API configuration and try again.")


if __name__ == '__main__':
    main()
