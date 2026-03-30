import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import bcrypt

from database import init_db, add_user, get_user, save_log, get_recent_logs, get_all_logs, get_streak, get_user_conditions, clear_user_logs
from ml_model import predict_health_score
from llm_tips import get_ai_insights, generate_interactive_html_report

# --- Page Config ---
st.set_page_config(page_title="Vitality AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

init_db()

# --- Cached LLM Calls ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_insight(user_id, streak, log_hash_string):
    df = get_recent_logs(user_id, 1)
    if not df.empty:
        return get_ai_insights(df.iloc[-1].to_dict(), streak)
    return ""

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_interactive_report(user_id, total_logs):
    """
    Generate an HTML report using Gemini based on all user logs.
    Cache validates on total_logs count so missing records bust it.
    """
    df = get_all_logs(user_id)
    if not df.empty:
        return generate_interactive_html_report(df)
    return "No data found."

# --- Routing State ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Landing'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# --- Dark/Light Mode State ---
if 'theme_is_dark' not in st.session_state:
    st.session_state.theme_is_dark = False

# --- Helpers ---
def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

def login_guest():
    user = get_user('Guest')
    if user:
        st.session_state.user_id = user[0]
        st.session_state.username = user[1]
        navigate_to('App')

# --- Theme Injector ---
def inject_theme():
    if st.session_state.theme_is_dark:
        st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #FFFFFF; }
                [data-testid="stHeader"] { background-color: #0E1117; }
                [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333;}
                [data-testid="stForm"] { background-color: #262730; border: 1px solid #444;}
                p, h1, h2, h3, h4, span, label, div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
                div.stButton > button:first-child { background-color: #333; color: white; border: 1px solid #555; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] { background-color: #FAFAFA; color: #000000; }
                [data-testid="stHeader"] { background-color: #FAFAFA; }
                [data-testid="stSidebar"] { background-color: #E8ECEF; border-right: 1px solid #DDD;}
                [data-testid="stForm"] { background-color: #FFFFFF; border: 1px solid #DDD;}
                p, h1, h2, h3, h4, span, label, div[data-testid="stMetricValue"] { color: #111111 !important; }
            </style>
        """, unsafe_allow_html=True)

inject_theme()

# ==========================================
# 1. LANDING PAGE
# ==========================================
if st.session_state.current_page == 'Landing':
    
    col_t1, col_t2 = st.columns([8, 1])
    with col_t2:
        st.session_state.theme_is_dark = st.toggle("🌙 Dark Mode", st.session_state.theme_is_dark)

    st.write("")
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🌿 Vitality AI")
        st.subheader("Your AI-Powered Personal Health Assistant.")
        st.markdown("""
        **Vitality AI** empowers you to take control of your well-being.
        - 📊 Track your Daily Vitality Score
        - 🤖 Learn your unique baseline with Machine Learning
        - 💧 Monitor sleep, exercise, hydration, and nutrition
        - ✨ Get dynamic emotional support & health tips fueled by Google Gemini!
        """)
        st.write("")
        if st.button("Get Started Now", type="primary", use_container_width=True):
            navigate_to('Auth')
    st.stop()

# ==========================================
# 2. AUTHENTICATION PAGE
# ==========================================
elif st.session_state.current_page == 'Auth':
    col_t1, col_t2 = st.columns([8, 1])
    with col_t2:
        st.session_state.theme_is_dark = st.toggle("🌙 Dark Mode", st.session_state.theme_is_dark)

    if st.button("← Back to Home"):
        navigate_to('Landing')
    st.title("Welcome Back")
    st.markdown("Please log in or create a simple account to continue.")
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(border=True):
            st.subheader("Login")
            with st.form("login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                submitted_login = st.form_submit_button("Login", type="primary", use_container_width=True)
                if submitted_login:
                    if len(l_user) > 0 and len(l_pass) > 0:
                        user = get_user(l_user)
                        if user and bcrypt.checkpw(l_pass.encode('utf-8'), user[2].encode('utf-8')):
                            st.session_state.user_id = user[0]
                            st.session_state.username = user[1]
                            navigate_to('App')
                        else:
                            st.error("Invalid username or password.")
                    else:
                        st.error("Please fill in both fields.")
    with col2:
        with st.container(border=True):
            st.subheader("Create Account")
            with st.form("signup_form"):
                s_user = st.text_input("Username")
                s_pass = st.text_input("Password", type="password")
                submitted_signup = st.form_submit_button("Sign Up", use_container_width=True)
                if submitted_signup:
                    if len(s_user) > 0 and len(s_pass) > 0: 
                        h = bcrypt.hashpw(s_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        if add_user(s_user, h):
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Username already exists.")
                    else:
                        st.error("Please fill in both fields.")
    st.markdown("---")
    st.subheader("Just want to explore?")
    if st.button("Continue as Guest", use_container_width=False):
        login_guest()
    st.stop()

# ==========================================
# 3. MAIN DASHBOARD APP
# ==========================================
elif st.session_state.current_page == 'App':
    if st.session_state.user_id is None:
        navigate_to('Auth')
        
    user_id = st.session_state.user_id

    st.sidebar.title(f"🌿 Vitality AI")
    st.session_state.theme_is_dark = st.sidebar.toggle("🌙 Toggle Dark Mode", st.session_state.theme_is_dark)
    
    st.sidebar.markdown(f"**Welcome, {st.session_state.username}!**")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Log Daily Health", "Medical Advice"])
    st.sidebar.markdown("---")
    streak = get_streak(user_id)
    st.sidebar.markdown(f"🔥 **Current Streak:** {streak} Days")
    if streak > 3:
        st.sidebar.success("Amazing consistency! Keep it up.")
    st.sidebar.markdown("---")
    st.sidebar.warning("⚠️ **Medical Disclaimer**\n\nThe information provided by this app is educational only.")
    
    if st.sidebar.button("Logout"):
        # WIPE GUEST DATA ON LOGOUT
        if st.session_state.username == 'Guest':
            clear_user_logs(st.session_state.user_id)
            
        st.session_state.user_id = None
        st.session_state.username = None
        navigate_to('Landing')

    # Download AI INTERACTIVE HTML DATA Button
    all_data = get_all_logs(user_id)
    if not all_data.empty:
        st.sidebar.markdown("### ✨ AI Data Export")
        st.sidebar.markdown("Convert your logs into a complete interactive HTML report.")
        
        # We wait for user to click button then dynamically provide download
        # Generating proactively would cost API calls on every sidebar load.
        # Instead, we will add a generation trigger:
        if st.sidebar.button("Generate AI HTML Report"):
            with st.spinner("Gemini is creating your interactive report..."):
                html_resp = fetch_interactive_report(user_id, len(all_data))
                
            st.sidebar.download_button(
                "📥 Click to Download HTML", 
                data=html_resp.encode('utf-8'), 
                file_name="Vitality_Report.html", 
                mime="text/html"
            )

    if menu == "Dashboard":
        st.title("📊 Health Dashboard")
        df = get_recent_logs(user_id, 30)
        
        if df.empty:
            st.info("No data logged yet. Head over to **Log Daily Health** to unlock your dashboard!")
        else:
            latest = df.iloc[-1]
            
            with st.container(border=True):
                st.subheader("✨ Gemini AI Insights")
                log_hash = f"{latest['log_date']}_{latest['health_score']}"
                with st.spinner("Gemini is analyzing today's metrics..."):
                    tip = fetch_insight(user_id, streak, log_hash)
                st.markdown(tip)

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container(border=True):
                    st.metric("Latest Health Score", f"{latest['health_score']:.1f}/100", delta="Great" if latest['health_score'] > 75 else None)
            with col2:
                with st.container(border=True):
                    st.metric("Sleep", f"{latest['sleep_hours']} hrs")
            with col3:
                with st.container(border=True):
                    st.metric("Exercise", f"{latest['exercise_minutes']} mins")
            with col4:
                with st.container(border=True):
                    st.metric("Water", f"{latest['water_intake']} L")

            st.markdown("---")
            st.subheader("📈 Core Vitality Trends")
            fig = go.Figure()
            
            text_color = "#FFFFFF" if st.session_state.theme_is_dark else "#000000"
            
            fig.add_trace(go.Scatter(
                x=df['log_date'], y=df['health_score'],
                name="Health Score", line=dict(color='#10B981', width=3),
                mode='lines+markers', marker=dict(size=8, symbol='circle')
            ))
            fig.add_trace(go.Bar(
                x=df['log_date'], y=df['exercise_minutes'],
                name="Exercise (mins)", marker_color='rgba(16, 185, 129, 0.4)',
                yaxis='y2'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=text_color),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
                yaxis=dict(title="Health Score", range=[0, 100], showgrid=False, color=text_color),
                yaxis2=dict(title="Exercise Minutes", overlaying='y', side='right', showgrid=False, color=text_color),
                xaxis=dict(color=text_color),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💧 Hydration Tracker")
                fig_water = px.area(df, x="log_date", y="water_intake", color_discrete_sequence=['#3B82F6'])
                fig_water.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=text_color), height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_water, use_container_width=True)
            with c2:
                st.subheader("🍽️ Meal Quality Split")
                meal_counts = df['meal_type'].value_counts()
                fig_pie = px.pie(values=meal_counts.values, names=meal_counts.index, hole=0.4)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=text_color), height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

            if len(all_data) >= 5:
                st.success("🚀 **AI Retraining Active:** Based on your history, the AI is now dynamically fine-tuning predictions to your specific baseline!")

    elif menu == "Log Daily Health":
        st.title("📝 Daily Health Check-in")
        st.markdown("Input your metrics. Over time, the AI will learn your baseline.")
        conds = get_user_conditions(user_id)
        
        with st.form("health_form", border=True):
            log_date = st.date_input("Date", date.today())
            col1, col2 = st.columns(2)
            with col1:
                sleep = st.slider("Sleep Hours", 0.0, 24.0, 7.5, 0.5)
                meal = st.selectbox("Meal Quality", ["Junk", "Average", "Healthy"], 1)
                water = st.number_input("Water Intake (Liters)", 0.0, 10.0, 2.0, 0.25)
            with col2:
                exercise = st.slider("Exercise (Minutes)", 0, 300, 30, 5)
                mood = st.selectbox("Mood", ["Poor", "Okay", "Good", "Great"], 2)
            st.markdown("---")
            st.subheader("Medical Profile Updates")
            c1, c2, c3 = st.columns(3)
            diab = c1.checkbox("Diabetes", value=conds['diabetes'])
            obesi = c2.checkbox("Obesity", value=conds['obesity'])
            hyper = c3.checkbox("Hypertension", value=conds['hypertension'])
            submitted = st.form_submit_button("Save & Predict Score", type="primary", use_container_width=True)

            if submitted:
                with st.spinner("AI is personalizing your metrics..."):
                    score = predict_health_score(user_id, sleep, exercise, meal, mood, water, diab, obesi, hyper)
                saved = save_log(user_id, str(log_date), sleep, exercise, meal, mood, water, diab, obesi, hyper, score)
                if saved:
                    st.success(f"Log saved successfully! Your predicted Health Score is **{score:.1f}/100**.")
                    st.balloons()
                else:
                    st.error("Failed to save log. Please try again.")

    elif menu == "Medical Advice":
        st.title("🩺 Personalized Guidance")
        conds = get_user_conditions(user_id)
        has_condition = False
        if conds['diabetes']:
            has_condition = True
            with st.expander("🩸 Managing Diabetes", expanded=True):
                st.markdown("- Focus on whole grains\n- Regular aerobic exercise\n- Monitor glycemic index")
        if conds['obesity']:
            has_condition = True
            with st.expander("⚖️ Managing Obesity", expanded=True):
                st.markdown("- Practice portion control\n- Incorporate cardio and strength training")
        if conds['hypertension']:
            has_condition = True
            with st.expander("❤️ Managing Hypertension", expanded=True):
                st.markdown("- Reduce sodium intake\n- Potassium-rich foods (DASH Diet)\n- Manage stress")
        if not has_condition:
            st.info("No pre-existing conditions registered. Keep up a balanced lifestyle!")
