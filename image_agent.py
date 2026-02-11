"""
Image Question Answering Agent
图像问答智能体

Standalone agent for image analysis and product identification using OpenAI GPT-4 Vision.
使用OpenAI GPT-4 Vision的独立图像分析和产品识别智能体。
"""

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import os
import base64
from io import BytesIO

# 加载环境变量 Load environment variables
load_dotenv()

# 获取API密钥 Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# 设置Serper密钥 Set Serper key
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY


def image_to_base64(image):
    """
    将PIL图像转换为base64编码
    Convert PIL Image to base64 encoding
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64 encoded image string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def analyze_image(image, prompt):
    """
    使用GPT-4 Vision分析图像
    Analyze image using GPT-4 Vision
    
    Args:
        image: PIL Image object
        prompt: Analysis prompt
        
    Returns:
        str: Analysis result
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        max_tokens=1000
    )
    
    # 转换图像 Convert image
    image_base64 = image_to_base64(image)
    
    # 构建消息 Build message
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    )
    
    # 获取响应 Get response
    response = llm.invoke([message])
    return response.content


def main():
    """主函数 Main function"""
    st.title("📸 Image Question Answering Agent")
    st.write("Upload an image and ask questions about products in it")
    
    # 验证API密钥 Validate API key
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found! Please add it to your .env file.")
        return
    
    # 文件上传 File uploader
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # 显示图像 Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # 分析图像 Analyze image
        with st.spinner("🔍 Analyzing image with GPT-4 Vision..."):
            try:
                analysis_prompt = """
                Analyze this image and identify all visible products.
                Provide:
                1. Product names and descriptions
                2. Brands (if identifiable)
                3. Product categories
                4. Notable features
                """
                
                image_analysis = analyze_image(image, analysis_prompt)
                
                # 显示分析结果 Display analysis
                st.write("---")
                st.write("### 🔍 Image Analysis")
                st.write(image_analysis)
                
                # 创建LLM和工具 Create LLM and tools
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.7,
                    api_key=OPENAI_API_KEY,
                    max_tokens=2000
                )
                
                serper_tool = SerperDevTool()
                
                # 创建智能体 Create agent
                image_agent = Agent(
                    role="Image Analysis and Recommendation Expert",
                    goal="Answer questions about products in images and provide recommendations",
                    backstory=f"""You are an expert at analyzing product images and helping users 
                    find similar products. You use web search to find current information and provide
                    detailed recommendations with prices and purchase links.
                    
                    Image Analysis Result: {image_analysis}
                    """,
                    tools=[serper_tool],
                    verbose=True,
                    llm=llm,
                    allow_delegation=True
                )
                
                # 用户问题 User questions
                st.write("---")
                questions = st.text_input(
                    "Ask questions about the image",
                    placeholder="Example: Where can I buy this? What's the price?"
                )
                
                if questions:
                    # 创建任务 Create task
                    qa_task = Task(
                        description=f"""
                        Based on the image analysis, answer: "{questions}"
                        
                        Provide:
                        1. Direct answer to the question
                        2. Product recommendations similar to the image
                        3. Where to buy (websites/stores)
                        4. Approximate prices
                        5. Purchase links if available
                        
                        Use web search for current information.
                        """,
                        agent=image_agent,
                        tools=[serper_tool],
                        expected_output="Comprehensive answer with recommendations and purchase information."
                    )
                    
                    # 创建团队 Create crew
                    crew = Crew(
                        agents=[image_agent],
                        tasks=[qa_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    # 执行 Execute
                    with st.spinner("🤖 AI is working on your question..."):
                        try:
                            result = crew.kickoff()
                            
                            # 显示结果 Display result
                            st.write("---")
                            st.write("### ✅ Answer")
                            st.write(str(result))
                            
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            
            except Exception as e:
                st.error(f"Image analysis failed: {str(e)}")
                st.info("Please make sure the image is valid and try again.")


if __name__ == '__main__':
    main()
