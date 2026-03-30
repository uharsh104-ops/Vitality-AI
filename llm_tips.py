import google.generativeai as genai
import textwrap

# Hardcoded for the requested integration task
GEMINI_API_KEY = "AIzaSyACNerPkygd031dASuAuDh4VNt-DyqrjNQ"
genai.configure(api_key=GEMINI_API_KEY)

def get_ai_insights(latest_log, streak):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        conditions = []
        if latest_log.get('diabetes', 0) == 1: conditions.append("Diabetes")
        if latest_log.get('obesity', 0) == 1: conditions.append("Obesity")
        if latest_log.get('hypertension', 0) == 1: conditions.append("Hypertension")
        condition_str = ", ".join(conditions) if conditions else "None"
        
        prompt = f"""
        You are 'Vitality AI', an empathetic, encouraging, and knowledgeable personal health assistant. 
        Your goal is to provide a brief paragraph (max 3 sentences) of emotional support and one actionable 
        health tip based on the user's metrics from today.
        
        Today's Metrics:
        - Health Score: {latest_log.get('health_score', 'N/A')}/100
        - Sleep: {latest_log.get('sleep_hours', 'N/A')} hours
        - Exercise: {latest_log.get('exercise_minutes', 'N/A')} minutes
        - Hydration: {latest_log.get('water_intake', 'N/A')} Liters
        - Prevailing Mood: {latest_log.get('mood', 'Average')}
        - Formally Tracked Conditions: {condition_str}
        - Current Logging Streak: {streak} days
        
        Provide your response in Markdown format. Be exceptionally kind and focus on the brightest spots.
        Limit your answer to 3 to 4 sentences total.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"*(AI Insights are currently unavailable. Error: {str(e)})*"

def generate_interactive_html_report(df):
    """
    Creates a dynamic, fully-contained HTML report from the user's DataFrame history using Gemini.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro') # Using pro for better formatting instructions
        csv_data = df.to_csv(index=False)
        
        prompt = f"""
        You are an expert web developer and data analyst. I am providing you with a user's health tracking history in CSV format.
        
        Your task is to generate ONE complete, fully standalone HTML file (including internal CSS and simple JS if necessary) that acts as an "Interactive AI Health Report" for the user.
        
        Requirements for the HTML file:
        1. It must look beautiful, modern, and utilize a sleek dark or light mode theme.
        2. Give an empathetic AI summary paragraph at the top analyzing their trends.
        3. Display their data in aesthetically pleasing HTML tables or custom CSS stat cards for averages (Sleep, Health Score, Hydration).
        4. Focus heavily on visual presentation.
        5. DO NOT wrap the output in markdown code blocks (like ```html). Output ONLY the raw HTML string starting with <!DOCTYPE html>.
        
        Here is the data:
        {csv_data}
        """
        
        response = model.generate_content(prompt)
        # Strip potential markdown blocks if Gemini stubbornly includes them
        raw_html = response.text.strip()
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
            
        return raw_html.strip()
    except Exception as e:
        return f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"
