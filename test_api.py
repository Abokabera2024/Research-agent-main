#!/usr/bin/env python3
"""
Quick test for OpenRouter API configuration
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def test_openrouter_api():
    """Test OpenRouter API connection."""
    load_dotenv()
    
    print("Testing OpenRouter API configuration...")
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {openrouter_key[:10]}...{openrouter_key[-4:]}")
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="openai/gpt-oss-20b:free",
            temperature=0,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-username/research-agent",
                "X-Title": "Research Agent"
            }
        )
        
        print("✅ LLM initialized successfully")
        
        # Test simple query
        response = llm.invoke("Hello, can you respond with 'API working'?")
        print(f"✅ API Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openrouter_api()
    if success:
        print("\n🎉 OpenRouter API is working correctly!")
    else:
        print("\n⚠️ API test failed - check your configuration")
