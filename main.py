"""
Multi-Agent LLM Product Recommendation

This application uses OpenAI's GPT models with CrewAI framework to provide:
1. Product recommendations based on user queries
2. Image analysis and product identification
3. Web search for product information
"""

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import os
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# ========================================
# Configuration
# ========================================

# Country-Currency mapping
COUNTRY_CURRENCY = {
    "Eurozone": "EUR",
    "China": "CNY",
}

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set Serper API key
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# Validate OpenAI API key
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found! Please add it to your .env file.")
    st.stop()

# ========================================
# Helper Functions
# ========================================

def image_to_base64(image):
    """
    Convert PIL Image to base64 encoded string for API transmission
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64 encoded image with data URI prefix
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_llm(model_name="gpt-4o-mini", temperature=0.7):
    """
    Create configured OpenAI LLM instance
    
    Args:
        model_name: OpenAI model name (gpt-4o, gpt-4o-mini, etc.)
        temperature: Creativity level (0-1, higher = more creative)
        
    Returns:
        ChatOpenAI: Configured LLM instance
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        max_tokens=2000
    )


def analyze_image_with_vision(image, prompt):
    """
    Analyze image using GPT-4 Vision API
    
    Args:
        image: PIL Image object
        prompt: Analysis prompt text
        
    Returns:
        str: Analysis result from GPT-4 Vision
    """
    from langchain_core.messages import HumanMessage
    
    # Use GPT-4o for vision analysis
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        max_tokens=1000
    )
    
    # Convert image to base64
    image_base64 = image_to_base64(image)
    
    # Build message with image
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    )
    
    # Get response from API
    response = llm.invoke([message])
    return response.content


# ========================================
# Streamlit UI Setup
# ========================================

st.set_page_config(
    page_title="AI Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI-Powered Product Recommendation")
st.markdown("### Intelligent Shopping Assistant with OpenAI GPT-4")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")

selected_country = st.sidebar.selectbox(
    "Select Your Region",
    list(COUNTRY_CURRENCY.keys())
)
currency = COUNTRY_CURRENCY[selected_country]

model_type = st.sidebar.selectbox(
    "Choose Model Type",
    (
        "Product Recommendation",
        "Image Question Answering",
        "Web Search"
    )
)

# Display current settings
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Region:** {selected_country}")
st.sidebar.markdown(f"**Currency:** {currency}")

# ========================================
# Model 1: Product Recommendation
# ========================================

if model_type == "Product Recommendation":
    st.write("---")
    st.subheader("💡 Product Recommendation")
    st.write("Tell us what you're looking for, and we'll recommend the best products.")
    
    # Create LLM instance
    llm = create_llm(model_name="gpt-4o-mini", temperature=0.7)
    
    # Initialize tools
    serper_tool = SerperDevTool()
    
    # Create recommendation agent
    recommendation_agent = Agent(
        role="Product Recommendation Specialist",
        goal="Provide personalized product recommendations based on user preferences and market trends.",
        backstory=f"""You are an expert product recommendation agent with extensive knowledge 
        across all product categories. You analyze user needs, search the web for latest information,
        and provide logical, convincing recommendations with detailed specifications, prices in {currency},
        and purchase links available in {selected_country}.""",
        tools=[serper_tool],
        verbose=True,
        llm=llm,
        allow_delegation=True
    )
    
    # User input
    text_prompt = st.text_input(
        "What product are you looking for?",
        placeholder="Example: I need a laptop for designer under 1500 EUR"
    )
    
    if text_prompt:
        st.write(f"**Your Query:** {text_prompt}")
        
        # Create recommendation task
        recommendation_task = Task(
            description=f"""
            Provide comprehensive product recommendations for: "{text_prompt}"
            
            Requirements:
            1. Search the web for current, accurate product information
            2. Recommend 3-5 specific products with detailed specifications
            3. Include brand names, model numbers, and key features
            4. Provide approximate prices in {currency}
            5. Suggest where to buy in {selected_country} with direct links
            6. Explain why each product is recommended
            7. Compare pros and cons of each option
            
            Focus on products currently available in {selected_country}.
            """,
            agent=recommendation_agent,
            tools=[serper_tool],
            expected_output="Detailed product recommendations with specifications, prices, purchase links, and explanations."
        )
        
        # Create execution crew
        crew = Crew(
            agents=[recommendation_agent],
            tasks=[recommendation_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute task
        with st.spinner("🤖 AI is analyzing and searching for the best products..."):
            try:
                result = crew.kickoff()
                result_str = str(result)
                
                # Display results
                st.write("---")
                st.subheader("✅ Recommendations")
                st.write(result_str)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API keys and try again.")


# ========================================
# Model 2: Image Question Answering
# ========================================

elif model_type == "Image Question Answering":
    st.write("---")
    st.subheader("📸 Image Analysis & Product Recognition")
    st.write("Upload a product image and ask questions about it.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze image with GPT-4 Vision
        with st.spinner("🔍 Analyzing image with GPT-4 Vision..."):
            try:
                analysis_prompt = """
                Analyze this image and identify all products visible. 
                Provide:
                1. Product names and descriptions
                2. Brands (if identifiable)
                3. Product categories
                4. Notable features or characteristics
                5. Estimated price range if recognizable
                """
                image_analysis = analyze_image_with_vision(image, analysis_prompt)
                
                # Display analysis
                st.write("---")
                st.subheader("🔍 Image Analysis Result")
                st.write(image_analysis)
                
                # Create LLM and tools
                llm = create_llm(model_name="gpt-4o", temperature=0.7)
                serper_tool = SerperDevTool()
                
                # Create image analysis agent
                image_agent = Agent(
                    role="Image Analysis and Product Recommendation Expert",
                    goal="Answer questions about products in images and provide purchase recommendations.",
                    backstory=f"""You are an expert at analyzing product images and providing recommendations.
                    Based on the image analysis, you help users find similar products or answer specific questions.
                    You search the web for current information and provide prices in {currency} with purchase 
                    links available in {selected_country}.
                    
                    Image Analysis: {image_analysis}
                    """,
                    tools=[serper_tool],
                    verbose=True,
                    llm=llm,
                    allow_delegation=True
                )
                
                # User question input
                st.write("---")
                questions = st.text_input(
                    "Ask questions about the image",
                    placeholder="Example: Where can I buy similar products? What's the price range?"
                )
                
                if questions:
                    # Create Q&A task
                    qa_task = Task(
                        description=f"""
                        Based on the image analysis, answer the user's question: "{questions}"
                        
                        Provide:
                        1. Direct answer to the question
                        2. Product recommendations similar to those in the image
                        3. Where to buy (websites/stores in {selected_country})
                        4. Approximate prices in {currency}
                        5. Direct purchase links if available
                        6. Product comparisons if relevant
                        
                        Use web search to find current, accurate information.
                        """,
                        agent=image_agent,
                        tools=[serper_tool],
                        expected_output="Comprehensive answer with product recommendations, links, and prices."
                    )
                    
                    # Create crew
                    crew = Crew(
                        agents=[image_agent],
                        tasks=[qa_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    # Execute
                    with st.spinner("🤖 AI is working on your question..."):
                        try:
                            result = crew.kickoff()
                            result_str = str(result)
                            
                            # Display result
                            st.write("---")
                            st.subheader("✅ Answer")
                            st.write(result_str)
                            
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            
            except Exception as e:
                st.error(f"Image analysis failed: {str(e)}")
                st.info("Please make sure the image is valid and try again.")


# ========================================
# Model 3: Web Search
# ========================================

elif model_type == "Web Search":
    st.write("---")
    st.subheader("🔍 Web Search")
    st.write("Search the web for product information, reviews, and comparisons.")
    
    # Create LLM and tools
    llm = create_llm(model_name="gpt-4o-mini", temperature=0.3)
    serper_tool = SerperDevTool()
    
    # Create search agent
    search_agent = Agent(
        role="Web Search Specialist",
        goal="Search and gather comprehensive product information from the web.",
        backstory=f"""You are an expert web search agent who finds accurate, relevant information.
        You know how to use search effectively and provide comprehensive answers with credible sources.
        When searching for products, you always include:
        - Purchase links for {selected_country}
        - Prices in {currency}
        - Reputable sellers and stores
        - Product reviews and comparisons
        """,
        tools=[serper_tool],
        verbose=True,
        llm=llm,
        allow_delegation=True
    )
    
    # User search input
    search_query = st.text_input(
        "Enter your search query",
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
            4. If product-related: prices in {currency}, purchase links for {selected_country}
            5. Product comparisons if multiple options exist
            6. User reviews and ratings
            7. Summary of main points
            
            Use web search to find current, accurate information.
            """,
            agent=search_agent,
            tools=[serper_tool],
            expected_output="Comprehensive search results with detailed information, sources, and relevant links."
        )
        
        # Create crew
        crew = Crew(
            agents=[search_agent],
            tasks=[search_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute search
        with st.spinner("🔍 Searching the web..."):
            try:
                result = crew.kickoff()
                result_str = str(result)
                
                # Display results
                st.write("---")
                st.subheader("✅ Search Results")
                st.write(result_str)
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.info("Please check your API keys and try again.")


# ========================================
# Footer 
# ========================================

st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Powered by OpenAI GPT-4 & CrewAI</strong></p>
    <p>Multi-Agent AI System for Intelligent Product Recommendations</p>
</div>
""", unsafe_allow_html=True)
