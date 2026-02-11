"""
Product Recommendation Agent

Standalone agent for product recommendations using OpenAI and CrewAI.
"""

import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
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
    st.title("🛍️ Product Recommendation Agent")
    st.write("Get personalized product recommendations powered by AI")
    
    # Validate API key
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found! Please add it to your .env file.")
        return
    
    # Create LLM instance
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        max_tokens=2000
    )
    
    # Initialize search tool
    serper_tool = SerperDevTool()
    
    # Create recommendation agent
    recommendation_agent = Agent(
        role="Product Recommendation Specialist",
        goal="Provide personalized product recommendations based on user preferences",
        backstory="""You are an expert product recommendation agent with deep knowledge 
        of products across all categories. You analyze user needs and provide logical,
        convincing recommendations. You always search for the latest information using
        available tools and suggest specific products with details.""",
        tools=[serper_tool],
        verbose=True,
        llm=llm,
        allow_delegation=True
    )
    
    # User input
    st.write("### Enter Your Product Query")
    text_prompt = st.text_input(
        "What are you looking for?",
        placeholder="Example: I need a laptop for programming under 1000 EUR"
    )
    
    if text_prompt:
        st.write(f"**Your Query:** {text_prompt}")
        
        # Create recommendation task
        recommendation_task = Task(
            description=f"""
            Provide the best product recommendations for: "{text_prompt}"
            
            Requirements:
            - Search the web for current product information
            - Recommend 3-5 specific products with details
            - Include brand names, models, key features
            - Provide approximate prices
            - Suggest where to buy with links
            - Explain why each product is recommended
            """,
            agent=recommendation_agent,
            tools=[serper_tool],
            expected_output="Detailed product recommendations with specifications, prices, and purchase information."
        )
        
        # Create crew
        crew = Crew(
            tasks=[recommendation_task],
            agents=[recommendation_agent],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute recommendation
        with st.spinner("🤖 AI is analyzing and searching for products..."):
            try:
                result = crew.kickoff()
                
                # Display results
                st.write("---")
                st.write("### ✅ Recommendations")
                st.write(str(result))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API configuration and try again.")


if __name__ == '__main__':
    main()
