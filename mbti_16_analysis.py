import streamlit as st
import pandas as pd
import random
from datetime import datetime
from supabase import create_client
import pytz
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- 그래프 설정 개선 ---
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.stats import pearsonr
from datetime import timedelta
import time
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, spearmanr
import io

# --- UI/UX 개선을 위한 설정 ---
st.set_page_config(
    page_title="내 로봇의 MBTI 진단 툴",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일 적용
st.markdown("""
<style>
    /* 전체 테마 개선 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* 헤더 스타일 */
    .main .block-container h1 {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* 카드 스타일 */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
    }
    
    /* 모던 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* 특별한 액션 버튼 (등록, 결과보기 등) */
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        border-radius: 16px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #ff5252 0%, #ff7043 100%);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        transform: translateY(-3px) scale(1.03);
    }
    
    /* 성공 버튼 (녹색 계열) */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #43a047 0%, #388e3c 100%);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
    }
    
    /* 경고 버튼 (주황 계열) */
    .stButton > button[data-testid="baseButton-tertiary"] {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    .stButton > button[data-testid="baseButton-tertiary"]:hover {
        background: linear-gradient(135deg, #f57c00 0%, #ef6c00 100%);
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.4);
    }
    
    /* 모던 입력 필드 스타일 */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 14px;
        color: #333333 !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
        color: #333333 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
    }
    
    /* 모던 드롭다운 스타일 */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #333333 !important;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* 드롭다운 옵션 스타일 */
    .stSelectbox > div > div > div > div {
        color: #333333 !important;
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* 라벨 스타일 */
    .stTextInput > label, .stSelectbox > label {
        color: #333333 !important;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* 프로그레스 바 */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 다크모드 토글 */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .main .block-container h1 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 그래프 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# MBTI 색상 매핑 (개선된 색상)
mbti_colors = {
    'ENFJ': '#FF6B6B', 'ENTJ': '#4ECDC4', 'ENTP': '#45B7D1', 'ENFP': '#96CEB4',
    'ESFJ': '#FFEAA7', 'ESFP': '#DDA0DD', 'ESTJ': '#98D8C8', 'ESTP': '#F7DC6F',
    'INFJ': '#BB8FCE', 'INFP': '#85C1E9', 'INTJ': '#F8C471', 'INTP': '#82E0AA',
    'ISFJ': '#F1948A', 'ISFP': '#85C1E9', 'ISTJ': '#F7DC6F', 'ISTP': '#D7BDE2'
}

plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# --- Supabase 설정 ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 환경 변수가 없으면 로컬 모드로 실행
if not SUPABASE_URL or not SUPABASE_KEY:
    st.warning("환경변수 SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다. 로컬 모드로 실행됩니다.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 세션 상태 초기화 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {"gender":"남", "age_group":"20대", "job":"학생"}
if 'robot_id' not in st.session_state:
    st.session_state.robot_id = "로봇A"
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

# --- 사용자 인증 함수들 ---
def authenticate_user(username, password):
    """사용자 로그인 인증"""
    try:
        # 간단한 사용자 인증 (실제 환경에서는 해시된 비밀번호 사용)
        users_db = {
            "user1": "password1",
            "user2": "password2", 
            "admin": "admin123"
        }
        
        if username in users_db and users_db[username] == password:
            user_type = "admin" if username == "admin" else "user"
            return True, user_type
        return False, None
    except Exception as e:
        st.error(f"로그인 오류: {e}")
        return False, None

def register_user(username, password, confirm_password):
    """사용자 회원가입"""
    if password != confirm_password:
        return False, "비밀번호가 일치하지 않습니다."
    
    if len(password) < 6:
        return False, "비밀번호는 6자 이상이어야 합니다."
    
    # 실제 환경에서는 데이터베이스에 저장
    try:
        # Supabase에 사용자 정보 저장 (예시)
        if supabase:
            user_data = {
                "username": username,
                "password": password,  # 실제로는 해시화 필요
                "user_type": "user",
                "created_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
            }
            # supabase.table("users").insert(user_data).execute()
        return True, "회원가입이 완료되었습니다."
    except Exception as e:
        return False, f"회원가입 오류: {e}"

def logout_user():
    """사용자 로그아웃"""
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

# --- 로봇 관리 함수들 ---
def save_robot_to_db(user_id, robot_name, robot_description=""):
    """로봇 정보를 Supabase에 저장"""
    try:
        record = {
            "user_id": user_id,
            "robot_name": robot_name,
            "robot_description": robot_description,
            "created_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
        }
        supabase.table("user_robots").insert(record).execute()
        return True
    except Exception as e:
        if "does not exist" in str(e) or "404" in str(e):
            st.warning("데이터베이스 테이블이 아직 생성되지 않았습니다. 로컬에만 저장됩니다.")
            return True
        else:
            st.error(f"로봇 저장 실패: {e}")
            return False

def load_user_robots(user_id):
    """사용자의 로봇 목록을 Supabase에서 불러오기"""
    try:
        res = supabase.table("user_robots").select("*").eq("user_id", user_id).execute()
        if res.data:
            return [robot['robot_name'] for robot in res.data]
        return []
    except Exception as e:
        if "does not exist" in str(e) or "404" in str(e):
            st.info("데이터베이스 테이블이 아직 생성되지 않았습니다. 기본 로봇을 사용합니다.")
            return []
        else:
            st.error(f"로봇 목록 불러오기 실패: {e}")
            return []

def delete_robot_from_db(user_id, robot_name):
    """로봇 정보를 Supabase에서 삭제"""
    try:
        supabase.table("user_robots").delete().eq("user_id", user_id).eq("robot_name", robot_name).execute()
        return True
    except Exception as e:
        if "does not exist" in str(e) or "404" in str(e):
            st.warning("데이터베이스 테이블이 아직 생성되지 않았습니다. 로컬에서만 삭제됩니다.")
            return True
        else:
            st.error(f"로봇 삭제 실패: {e}")
            return False

# 초기 로봇 목록 로드
if 'robot_list' not in st.session_state:
    user_robots = load_user_robots(USER_ID)
    if user_robots:
        st.session_state.robot_list = user_robots
    else:
        st.session_state.robot_list = ["로봇A"]

# --- MBTI 16유형 HRI 안내 데이터 ---
def load_mbti_guide():
    return {
        "ENFJ": {
            "description": "리더십과 공감 능력을 겸비한 타입입니다. 다른 사람의 성장을 돕고, 팀의 조화를 중시합니다. 로봇과의 상호작용에서도 따뜻하고 격려적인 스타일을 선호하며, 의미 있는 대화와 함께 성장하는 경험을 추구합니다.",
            "examples": [
                "안녕하세요! 오늘 어떤 도움이 필요하신지 편하게 말씀해 주세요.",
                "함께 이 문제를 해결해보는 건 어떨까요? 당신의 생각이 궁금해요.",
                "진행 상황을 확인해보니 정말 잘 하고 계시네요! 더 발전할 수 있는 부분도 제안드릴게요.",
                "혹시 어려운 점이 있으시면 언제든 말씀해 주세요. 함께 찾아보겠습니다."
            ],
            "hri_style": "공감적 리더십, 격려와 성장 지향, 팀워크 중시"
        },
        "ENTJ": {
            "description": "전략적 사고와 효율성을 중시하는 타입입니다. 명확한 목표 설정과 체계적인 접근을 선호하며, 로봇과의 상호작용에서도 논리적이고 결과 지향적인 스타일을 추구합니다.",
            "examples": [
                "목표 달성을 위해 체계적으로 안내해 드리겠습니다. 단계별로 진행하시죠.",
                "현재 상황을 분석한 결과, 이 방법이 가장 효율적일 것 같습니다.",
                "시간을 절약하기 위해 핵심만 요약해서 알려드릴게요.",
                "다음 단계로 넘어가기 전에 현재 진행상황을 확인해보시겠어요?"
            ],
            "hri_style": "전략적 사고, 효율성 중시, 목표 지향적"
        },
        "ENTP": {
            "description": "창의적이고 혁신적인 사고를 가진 타입입니다. 새로운 아이디어와 다양한 가능성을 탐구하며, 로봇과의 상호작용에서도 자유롭고 창의적인 대화를 선호합니다.",
            "examples": [
                "흥미로운 새로운 접근법을 제안해드릴게요. 어떻게 생각하세요?",
                "이 문제를 다른 각도에서 바라보면 어떨까요?",
                "여러 가지 옵션이 있는데, 어떤 것이 가장 흥미로우신가요?",
                "함께 실험해보면서 새로운 해결책을 찾아보는 건 어떨까요?"
            ],
            "hri_style": "창의적 사고, 다양한 옵션 탐구, 실험적 접근"
        },
        "ENFP": {
            "description": "열정적이고 창의적인 타입입니다. 가능성과 새로운 경험을 추구하며, 로봇과의 상호작용에서도 진심 어린 격려와 함께 성장하는 경험을 중시합니다.",
            "examples": [
                "당신의 아이디어가 정말 흥미롭네요! 함께 발전시켜보는 건 어떨까요?",
                "새로운 경험을 함께 해보는 게 어떨까요? 당신의 생각이 궁금해요.",
                "정말 잘하고 계시네요! 더 멋진 결과를 만들어보시죠.",
                "함께 즐겁게 배워가면서 새로운 가능성을 발견해보아요."
            ],
            "hri_style": "열정적 격려, 창의적 협력, 성장 지향"
        },
        "ESFJ": {
            "description": "협력적이고 사회적인 타입입니다. 다른 사람들의 필요를 잘 파악하고 도움을 주며, 로봇과의 상호작용에서도 정중하고 세심한 배려를 선호합니다.",
            "examples": [
                "도움이 필요하시면 언제든 편하게 말씀해 주세요. 함께 해결해보겠습니다.",
                "모두가 편안하게 참여할 수 있도록 안내해드릴게요.",
                "궁금한 점이나 어려운 점이 있으시면 언제든 도와드릴게요.",
                "함께 협력해서 좋은 결과를 만들어보시죠."
            ],
            "hri_style": "협력적 배려, 사회적 조화, 정중한 지원"
        },
        "ESFP": {
            "description": "즉각적이고 실용적인 타입입니다. 현재의 경험을 중시하며, 로봇과의 상호작용에서도 즉각적이고 즐거운 경험을 선호합니다.",
            "examples": [
                "즉시 시작해보시죠! 재미있게 진행해보아요.",
                "필요하신 것이 있으면 바로 말씀해 주세요. 바로 도와드릴게요.",
                "지금 당장 해보시는 건 어떨까요? 함께 즐거운 시간을 만들어보아요.",
                "실용적으로 도움이 될 수 있는 방법을 제안해드릴게요."
            ],
            "hri_style": "즉각적 지원, 실용적 접근, 즐거운 경험"
        },
        "ESTJ": {
            "description": "체계적이고 책임감 있는 타입입니다. 명확한 규칙과 절차를 중시하며, 로봇과의 상호작용에서도 정확하고 체계적인 안내를 선호합니다.",
            "examples": [
                "정해진 절차에 따라 체계적으로 안내해드리겠습니다.",
                "규칙을 꼭 지켜주시면 더 효율적으로 진행할 수 있습니다.",
                "단계별로 정확하게 진행해보시죠. 각 단계를 명확히 안내해드릴게요.",
                "책임감 있게 완료할 수 있도록 도와드리겠습니다."
            ],
            "hri_style": "체계적 안내, 규칙 준수, 책임감 있는 접근"
        },
        "ESTP": {
            "description": "실용적이고 적응력 있는 타입입니다. 현재 상황에 맞춰 유연하게 대응하며, 로봇과의 상호작용에서도 실용적이고 즉각적인 해결책을 선호합니다.",
            "examples": [
                "바로 실행해보시는 건 어떨까요? 실용적인 방법을 제안해드릴게요.",
                "현재 상황에 맞춰 유연하게 대응해보시죠.",
                "즉시 실행할 수 있는 방법을 알려드릴게요.",
                "실용적으로 문제를 해결해보는 건 어떨까요?"
            ],
            "hri_style": "실용적 해결, 유연한 대응, 즉각적 실행"
        },
        "INFJ": {
            "description": "깊은 통찰력과 공감 능력을 가진 타입입니다. 의미 있는 관계와 깊이 있는 이해를 추구하며, 로봇과의 상호작용에서도 세심한 배려와 의미 있는 대화를 선호합니다.",
            "examples": [
                "당신의 마음을 이해합니다. 함께 의미 있는 경험을 만들어보아요.",
                "깊이 있는 대화를 통해 더 나은 해결책을 찾아보시죠.",
                "당신의 생각과 감정을 소중히 여기며 함께 성장해보겠습니다.",
                "의미 있는 결과를 만들어보기 위해 함께 노력해보아요."
            ],
            "hri_style": "깊은 공감, 의미 있는 관계, 세심한 배려"
        },
        "INFP": {
            "description": "가치와 감정을 중시하는 타입입니다. 진정성과 자기표현을 추구하며, 로봇과의 상호작용에서도 개인의 가치와 감정을 존중하는 스타일을 선호합니다.",
            "examples": [
                "당신의 감정과 가치를 소중히 생각합니다. 편하게 표현해주세요.",
                "진정성 있는 대화를 통해 함께 성장해보아요.",
                "당신만의 독특한 관점을 존중하며 함께 배워가보겠습니다.",
                "의미 있는 경험을 만들어보기 위해 당신의 생각을 들려주세요."
            ],
            "hri_style": "가치 존중, 진정성 있는 대화, 개성 중시"
        },
        "INTJ": {
            "description": "전략적 사고와 미래지향적 비전을 가진 타입입니다. 체계적 분석과 장기적 계획을 중시하며, 로봇과의 상호작용에서도 논리적이고 전략적인 접근을 선호합니다.",
            "examples": [
                "장기적인 비전을 고려한 전략을 제안해드릴게요.",
                "체계적으로 분석한 결과를 바탕으로 안내해드리겠습니다.",
                "미래 지향적인 접근법으로 문제를 해결해보시죠.",
                "전략적 사고를 통해 더 나은 결과를 만들어보겠습니다."
            ],
            "hri_style": "전략적 사고, 체계적 분석, 미래지향적 접근"
        },
        "INTP": {
            "description": "분석적 사고와 논리적 탐구를 중시하는 타입입니다. 복잡한 문제를 해결하고 새로운 아이디어를 탐구하며, 로봇과의 상호작용에서도 논리적이고 분석적인 접근을 선호합니다.",
            "examples": [
                "함께 논리적으로 분석해보면서 새로운 해결책을 찾아보시죠.",
                "복잡한 문제를 단계별로 분석해보는 건 어떨까요?",
                "새로운 관점에서 문제를 바라보면서 창의적 해결책을 찾아보아요.",
                "함께 탐구하면서 더 나은 방법을 발견해보겠습니다."
            ],
            "hri_style": "분석적 사고, 논리적 탐구, 창의적 해결"
        },
        "ISFJ": {
            "description": "조용하고 신뢰할 수 있는 타입입니다. 실질적 지원과 세심한 배려를 제공하며, 로봇과의 상호작용에서도 안정적이고 신뢰할 수 있는 스타일을 선호합니다.",
            "examples": [
                "필요하실 때 언제든 도와드릴게요. 편하게 이용하세요.",
                "안정적이고 신뢰할 수 있는 지원을 제공해드리겠습니다.",
                "세심하게 배려하며 함께 성장해보아요.",
                "실질적인 도움이 될 수 있도록 최선을 다하겠습니다."
            ],
            "hri_style": "안정적 지원, 신뢰할 수 있는 배려, 실질적 도움"
        },
        "ISFP": {
            "description": "온화하고 감성적인 타입입니다. 개인의 자유와 편안함을 중시하며, 로봇과의 상호작용에서도 부드럽고 개인을 존중하는 스타일을 선호합니다.",
            "examples": [
                "당신만의 방식과 속도를 존중하며 함께해보아요.",
                "편안하고 자유로운 분위기에서 진행해보시죠.",
                "당신의 감정과 필요를 소중히 생각합니다.",
                "부드럽고 따뜻한 마음으로 함께 성장해보겠습니다."
            ],
            "hri_style": "온화한 배려, 개인 존중, 자유로운 분위기"
        },
        "ISTJ": {
            "description": "정확하고 책임감 있는 타입입니다. 규칙과 절차를 중시하며, 로봇과의 상호작용에서도 정확하고 체계적인 안내를 선호합니다.",
            "examples": [
                "정확하고 체계적으로 안내해드리겠습니다.",
                "규정과 지침에 따라 정확하게 진행해보시죠.",
                "책임감 있게 완료할 수 있도록 도와드리겠습니다.",
                "단계별로 정확하게 진행하면서 결과를 만들어보아요."
            ],
            "hri_style": "정확한 안내, 규칙 준수, 책임감 있는 접근"
        },
        "ISTP": {
            "description": "실용적이고 적응력 있는 타입입니다. 간단하고 효율적인 해결책을 선호하며, 로봇과의 상호작용에서도 실용적이고 직접적인 접근을 선호합니다.",
            "examples": [
                "간단하고 실용적으로 도와드릴게요.",
                "필요하실 때 말씀만 하시면 바로 도와드리겠습니다.",
                "효율적으로 문제를 해결해보시죠.",
                "직접적이고 실용적인 방법으로 진행해보아요."
            ],
            "hri_style": "실용적 해결, 간결한 접근, 직접적 지원"
        }
    }
guide_data = load_mbti_guide()

# --- 고급 분석 함수들 ---
def calculate_mbti_correlations(df):
    """MBTI 유형 간 상관관계 분석"""
    # MBTI 유형을 원-핫 인코딩
    mbti_dummies = pd.get_dummies(df['mbti'])
    
    # 상관관계 계산
    correlations = mbti_dummies.corr()
    
    return correlations

def analyze_mbti_compatibility(user_mbti, robot_mbti):
    """MBTI 호환성 분석"""
    # MBTI 유형별 호환성 매트릭스 (간단한 예시)
    compatibility_matrix = {
        'ENFJ': {'ENFP': 0.9, 'INFJ': 0.8, 'ENFJ': 1.0, 'ENTJ': 0.7},
        'ENTJ': {'ENTP': 0.9, 'INTJ': 0.8, 'ENTJ': 1.0, 'ENFJ': 0.7},
        'ENFP': {'ENFJ': 0.9, 'INFP': 0.8, 'ENFP': 1.0, 'ENTP': 0.7},
        'ENTP': {'ENTJ': 0.9, 'INTP': 0.8, 'ENTP': 1.0, 'ENFP': 0.7},
        'INFJ': {'ENFJ': 0.8, 'INFP': 0.9, 'INFJ': 1.0, 'INTJ': 0.7},
        'INTJ': {'ENTJ': 0.8, 'INTP': 0.9, 'INTJ': 1.0, 'INFJ': 0.7},
        'INFP': {'ENFP': 0.8, 'INFJ': 0.9, 'INFP': 1.0, 'ISFP': 0.7},
        'INTP': {'ENTP': 0.8, 'INTJ': 0.9, 'INTP': 1.0, 'ISTP': 0.7},
        'ISFJ': {'ESFJ': 0.8, 'ISFP': 0.7, 'ISFJ': 1.0, 'ISTJ': 0.9},
        'ISFP': {'ESFP': 0.8, 'INFP': 0.7, 'ISFP': 1.0, 'ISFJ': 0.7},
        'ISTJ': {'ESTJ': 0.8, 'ISFJ': 0.9, 'ISTJ': 1.0, 'ISTP': 0.7},
        'ISTP': {'ESTP': 0.8, 'INTP': 0.7, 'ISTP': 1.0, 'ISTJ': 0.7},
        'ESFJ': {'ISFJ': 0.8, 'ESFP': 0.9, 'ESFJ': 1.0, 'ESTJ': 0.7},
        'ESFP': {'ISFP': 0.8, 'ENFP': 0.7, 'ESFP': 1.0, 'ESFJ': 0.9},
        'ESTJ': {'ISTJ': 0.8, 'ESFJ': 0.7, 'ESTJ': 1.0, 'ESTP': 0.9},
        'ESTP': {'ISTP': 0.8, 'ENTP': 0.7, 'ESTP': 1.0, 'ESTJ': 0.9}
    }
    
    if user_mbti in compatibility_matrix and robot_mbti in compatibility_matrix[user_mbti]:
        compatibility = compatibility_matrix[user_mbti][robot_mbti]
        if compatibility >= 0.9:
            level = "매우 높음"
            color = "🟢"
        elif compatibility >= 0.8:
            level = "높음"
            color = "🟡"
        elif compatibility >= 0.7:
            level = "보통"
            color = "🟠"
        else:
            level = "낮음"
            color = "🔴"
        
        return compatibility, level, color
    else:
        return 0.5, "알 수 없음", "⚪"

def create_progress_timeline(df, user_id):
    """진행률 타임라인 생성"""
    user_data = df[df['user_id'] == user_id].sort_values('timestamp')
    if len(user_data) < 2:
        return None
    
    # MBTI 변화 추적
    timeline_data = []
    for idx, row in user_data.iterrows():
        timeline_data.append({
            'date': row['timestamp'],
            'mbti': row['mbti'],
            'robot_id': row['robot_id']
        })
    
    return timeline_data

def generate_personalized_recommendations(df, user_id, robot_id):
    """개인화된 추천 생성"""
    recommendations = []
    
    # 사용자 패턴 분석
    user_data = df[(df['user_id'] == user_id) & (df['robot_id'] == robot_id)]
    if len(user_data) > 0:
        latest_mbti = user_data.iloc[-1]['mbti']
        
        # MBTI별 추천
        mbti_recommendations = {
            'ENFJ': "리더십과 공감 능력을 활용한 로봇 상호작용을 시도해보세요.",
            'ENTJ': "전략적 사고와 효율성을 중시하는 로봇 설정을 권장합니다.",
            'ENFP': "창의적이고 유연한 로봇 응답 방식을 선호할 것 같습니다.",
            'ENTP': "혁신적이고 논리적인 로봇 대화 스타일이 적합합니다.",
            'INFJ': "깊이 있는 이해와 직관적 상호작용을 추천합니다.",
            'INTJ': "체계적이고 미래지향적인 로봇 기능을 활용하세요.",
            'INFP': "가치와 감정을 중시하는 로봇 설정이 좋겠습니다.",
            'INTP': "분석적이고 탐구적인 로봇 상호작용을 시도해보세요.",
            'ISFJ': "안정적이고 실용적인 로봇 기능을 선호할 것 같습니다.",
            'ISFP': "자유롭고 감성적인 로봇 응답 방식을 권장합니다.",
            'ISTJ': "정확하고 체계적인 로봇 운영 방식을 추천합니다.",
            'ISTP': "실용적이고 즉흥적인 로봇 상호작용이 적합합니다.",
            'ESFJ': "협력적이고 사회적인 로봇 기능을 활용하세요.",
            'ESFP': "즉각적이고 즐거운 로봇 경험을 선호할 것 같습니다.",
            'ESTJ': "조직적이고 규칙적인 로봇 운영 방식을 권장합니다.",
            'ESTP': "실용적이고 적응력 있는 로봇 설정이 좋겠습니다."
        }
        
        if latest_mbti in mbti_recommendations:
            recommendations.append(mbti_recommendations[latest_mbti])
        
        # 진단 빈도 분석
        if len(user_data) > 1:
            avg_days = (pd.to_datetime(user_data['timestamp'].iloc[-1]) - 
                       pd.to_datetime(user_data['timestamp'].iloc[0])).days / (len(user_data) - 1)
            if avg_days > 7:
                recommendations.append("정기적인 진단을 통해 MBTI 변화를 추적해보세요.")
            else:
                recommendations.append("진단 간격을 조금 늘려서 더 안정적인 결과를 얻어보세요.")
    
    return recommendations

def create_animated_chart(data, chart_type="line"):
    """애니메이션 차트 생성"""
    if chart_type == "line":
        fig = px.line(data, x='date', y='value', 
                     title="애니메이션 트렌드",
                     color_discrete_map=mbti_colors)
        fig.update_layout(
            xaxis=dict(range=[data['date'].min(), data['date'].max()]),
            yaxis=dict(range=[data['value'].min(), data['value'].max()]),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="재생",
                    method="animate",
                    args=[None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True}]
                )]
            )]
        )
    return fig

def predict_mbti_trends(df, user_id, robot_id):
    """사용자의 MBTI 변화 트렌드 예측"""
    user_data = df[(df['user_id'] == user_id) & (df['robot_id'] == robot_id)].sort_values('timestamp')
    
    if len(user_data) < 3:
        return None, "예측을 위해서는 최소 3개의 진단 데이터가 필요합니다."
    
    # MBTI 유형을 숫자로 매핑
    mbti_types = sorted(df['mbti'].unique())
    mbti_map = {mbti: idx for idx, mbti in enumerate(mbti_types)}
    
    user_data['mbti_num'] = user_data['mbti'].map(mbti_map)
    
    # 간단한 선형 트렌드 예측
    x = np.arange(len(user_data))
    y = user_data['mbti_num'].values
    
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        next_prediction = slope * (len(x)) + intercept
        
        # 예측값을 MBTI 유형으로 변환
        predicted_idx = int(round(next_prediction))
        predicted_mbti = mbti_types[predicted_idx] if 0 <= predicted_idx < len(mbti_types) else mbti_types[-1]
        
        trend_direction = "상승" if slope > 0.1 else "하락" if slope < -0.1 else "안정"
        
        return predicted_mbti, f"다음 진단 예측: {predicted_mbti} (트렌드: {trend_direction})"
    
    return None, "예측할 수 있는 충분한 데이터가 없습니다."

def cluster_users_by_mbti(df):
    """MBTI 유형별 사용자 클러스터링"""
    # MBTI 유형을 원-핫 인코딩
    mbti_dummies = pd.get_dummies(df['mbti'])
    
    if len(mbti_dummies) < 3:
        return None, "클러스터링을 위해서는 최소 3개의 데이터가 필요합니다."
    
    # PCA로 차원 축소
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(mbti_dummies)
    
    # K-means 클러스터링
    n_clusters = min(5, len(scaled_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 결과를 데이터프레임에 추가
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    return df_clustered, f"{n_clusters}개의 클러스터로 그룹화 완료"

def create_mbti_network(df):
    """MBTI 유형 간 관계 네트워크 생성"""
    # MBTI 유형 간 공존 관계 계산
    mbti_pairs = []
    for _, group in df.groupby(['user_id', 'robot_id']):
        if len(group) > 1:
            mbti_list = group['mbti'].tolist()
            for i in range(len(mbti_list)-1):
                mbti_pairs.append((mbti_list[i], mbti_list[i+1]))
    
    if not mbti_pairs:
        return None, "네트워크를 생성할 수 있는 충분한 데이터가 없습니다."
    
    # 관계 빈도 계산
    pair_counts = {}
    for pair in mbti_pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # NetworkX 그래프 생성
    G = nx.Graph()
    for (mbti1, mbti2), weight in pair_counts.items():
        G.add_edge(mbti1, mbti2, weight=weight)
    
    return G, f"{len(G.nodes)}개 노드, {len(G.edges)}개 엣지의 네트워크 생성"

def generate_ai_insights(df, user_id, robot_id):
    """AI 기반 인사이트 생성"""
    insights = []
    
    # 사용자 데이터 분석
    user_data = df[(df['user_id'] == user_id) & (df['robot_id'] == robot_id)]
    
    if len(user_data) > 0:
        # 가장 최근 MBTI
        latest_mbti = user_data.iloc[-1]['mbti']
        insights.append(f"🎯 현재 {robot_id}의 MBTI: {latest_mbti}")
        
        # 변화 패턴 분석
        if len(user_data) > 1:
            mbti_list = user_data['mbti'].tolist()
            changes = []
            for i in range(1, len(mbti_list)):
                if mbti_list[i] != mbti_list[i-1]:
                    changes.append(f"{mbti_list[i-1]} → {mbti_list[i]}")
            
            if changes:
                change_counts = pd.Series(changes).value_counts()
                most_common_change = change_counts.index[0] if len(change_counts) > 0 else "변화 없음"
                insights.append(f"📈 주요 변화 패턴: {most_common_change}")
            else:
                insights.append("📈 주요 변화 패턴: 변화 없음")
        
        # 진단 빈도 분석
        avg_days_between = None
        if len(user_data) > 1:
            timestamps = pd.to_datetime(user_data['timestamp'])
            time_diffs = timestamps.diff().dropna()
            avg_days_between = time_diffs.mean().days
            insights.append(f"⏰ 평균 진단 간격: {avg_days_between:.1f}일")
    
    # 전체 데이터 기반 인사이트
    if len(df) > 0:
        most_common_mbti = df['mbti'].mode().iloc[0] if not df['mbti'].mode().empty else "N/A"
        insights.append(f"🏆 전체에서 가장 인기 있는 MBTI: {most_common_mbti}")
        
        # 연령대별 분석
        if 'age_group' in df.columns:
            age_mbti = df.groupby('age_group')['mbti'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
            for age, mbti in age_mbti.items():
                insights.append(f"👥 {age}대 선호 MBTI: {mbti}")
    
    return insights

def create_3d_mbti_chart(df):
    """3D MBTI 분포 차트"""
    if len(df) == 0:
        return None, "3D 차트를 생성할 수 있는 데이터가 없습니다."
    
    # MBTI 유형별, 성별, 연령대별 분포
    mbti_counts = df.groupby(['mbti', 'gender', 'age_group']).size().reset_index(name='count')
    
    # 3D 산점도 생성
    fig = px.scatter_3d(mbti_counts, 
                        x='mbti', y='gender', z='age_group', 
                        size='count', color='mbti',
                        title="3D MBTI 분포 분석",
                        color_discrete_map=mbti_colors)
    
    fig.update_layout(height=600)
    return fig, "3D 차트 생성 완료"

#### HRI 모델 목록
def load_hri_models():
    try:
        res = supabase.table("hri_models").select("*").execute()
        return res.data or []
    except Exception as e:
        st.error(f"HRI 모델 불러오기 오류: {e}")
        return []
hri_models = load_hri_models()

# --- 장소별 설문 질문 ---
def get_questions_by_location(location):
    """장소별 설문 질문 반환"""
    
    # 공통 기본 질문들
    base_questions = [
        {"id":"Q1","text":"로봇이 먼저 인사할 때 당신의 반응은?",
         "choices":["즉시 대화에 참여한다","잠시 상황을 관찰한다"],"axes":("E","I")},
        {"id":"Q2","text":"여러 사람과 로봇을 사용할 때 선호하는 방식은?",
         "choices":["모두가 함께 참여한다","개인적으로 1:1 상호작용한다"],"axes":("E","I")},
        {"id":"Q3","text":"로봇과 대화할 때 당신의 스타일은?",
         "choices":["적극적으로 질문하고 의견을 표현한다","로봇의 설명을 듣고 생각한 후 반응한다"],"axes":("E","I")},
        
        {"id":"Q4","text":"로봇의 안내 방식을 선호하는 스타일은?",
         "choices":["단계별로 구체적인 세부사항을 제공한다","전체적인 맥락과 의미를 먼저 설명한다"],"axes":("S","N")},
        {"id":"Q5","text":"새로운 로봇 기능을 배울 때 선호하는 방법은?",
         "choices":["실제로 직접 조작해보며 학습한다","개념과 원리를 먼저 이해한 후 시도한다"],"axes":("S","N")},
        {"id":"Q6","text":"로봇에게 작업을 요청할 때 선호하는 방식은?",
         "choices":["구체적이고 명확한 지시사항을 준다","일반적인 목표와 방향성만 제시한다"],"axes":("S","N")},
        
        {"id":"Q7","text":"로봇과 의사결정을 할 때 중시하는 것은?",
         "choices":["논리적 분석과 객관적 데이터","감정적 공감과 주관적 경험"],"axes":("T","F")},
        {"id":"Q8","text":"로봇이 실수를 했을 때 당신의 반응은?",
         "choices":["문제를 분석하고 해결책을 찾는다","로봇의 감정을 고려하여 대화한다"],"axes":("T","F")},
        {"id":"Q9","text":"로봇에게 피드백을 줄 때 선호하는 스타일은?",
         "choices":["정확하고 구체적인 개선점을 제시한다","긍정적 격려와 함께 조언한다"],"axes":("T","F")},
        
        {"id":"Q10","text":"로봇과의 일정 관리에서 선호하는 방식은?",
         "choices":["미리 계획하고 체계적으로 진행한다","상황에 따라 유연하게 조정한다"],"axes":("J","P")},
        {"id":"Q11","text":"로봇과 새로운 활동을 할 때의 접근법은?",
         "choices":["정해진 규칙과 절차를 따른다","즉흥적이고 창의적으로 시도한다"],"axes":("J","P")},
        {"id":"Q12","text":"로봇과의 상호작용 결과를 정리할 때 선호하는 방식은?",
         "choices":["명확한 요약과 결론을 도출한다","다양한 관점과 가능성을 제시한다"],"axes":("J","P")}
    ]
    
    # 장소별 특화 질문들
    location_specific_questions = {
        "병원": [
            {"id":"Q13","text":"진료 대기 중 로봇이 안내할 때 선호하는 방식은?",
             "choices":["상세한 진료 과정과 예상 시간을 알려준다","간단한 안내만 하고 조용히 기다린다"],"axes":("S","N")},
            {"id":"Q14","text":"로봇이 건강 정보를 제공할 때 당신의 반응은?",
             "choices":["의학적 전문 용어로 정확하게 설명한다","일반인이 이해하기 쉬운 용어로 설명한다"],"axes":("T","F")},
            {"id":"Q15","text":"응급 상황에서 로봇의 역할에 대한 생각은?",
             "choices":["즉시 의료진에게 연결하는 것이 중요하다","환자의 상태를 먼저 파악하고 대응한다"],"axes":("J","P")}
        ],
        "도서관": [
            {"id":"Q13","text":"도서 검색 로봇과 상호작용할 때 선호하는 방식은?",
             "choices":["키워드로 정확한 도서를 찾아준다","관련 도서들을 추천해준다"],"axes":("S","N")},
            {"id":"Q14","text":"도서관 내 로봇이 조용함을 유지하는 것에 대한 생각은?",
             "choices":["절대적으로 지켜야 하는 규칙이다","상황에 따라 적절히 조절한다"],"axes":("J","P")},
            {"id":"Q15","text":"로봇이 도서 추천을 할 때 당신의 반응은?",
             "choices":["개인 취향을 분석해서 추천한다","인기 도서나 베스트셀러를 추천한다"],"axes":("T","F")}
        ],
        "쇼핑몰": [
            {"id":"Q13","text":"상품 안내 로봇과 상호작용할 때 선호하는 방식은?",
             "choices":["원하는 상품을 정확히 찾아준다","다양한 옵션을 제시해준다"],"axes":("S","N")},
            {"id":"Q14","text":"로봇이 상품을 추천할 때 당신의 반응은?",
             "choices":["객관적인 상품 정보를 제공한다","개인 취향을 고려한 추천을 한다"],"axes":("T","F")},
            {"id":"Q15","text":"쇼핑 중 로봇의 적극적인 안내에 대한 생각은?",
             "choices":["필요할 때만 도움을 요청한다","적극적으로 상품을 소개해준다"],"axes":("E","I")}
        ],
        "학교": [
            {"id":"Q13","text":"수업 중 로봇이 도움을 줄 때 선호하는 방식은?",
             "choices":["개별적으로 질문에 답변한다","전체 학생에게 공통 설명을 한다"],"axes":("E","I")},
            {"id":"Q14","text":"로봇이 학습 진도를 관리할 때 당신의 반응은?",
             "choices":["체계적으로 학습 계획을 세워준다","유연하게 학습 속도를 조절한다"],"axes":("J","P")},
            {"id":"Q15","text":"로봇이 학습 성과를 평가할 때 선호하는 방식은?",
             "choices":["정확한 점수와 피드백을 제공한다","격려와 함께 개선점을 제시한다"],"axes":("T","F")}
        ],
        "공항": [
            {"id":"Q13","text":"체크인 로봇과 상호작용할 때 선호하는 방식은?",
             "choices":["빠르고 효율적으로 처리한다","친근하게 대화하며 진행한다"],"axes":("T","F")},
            {"id":"Q14","text":"로봇이 안내할 때 당신의 반응은?",
             "choices":["정확한 위치와 시간 정보를 제공한다","전체적인 맥락을 설명한다"],"axes":("S","N")},
            {"id":"Q15","text":"비행 지연 시 로봇의 대응에 대한 생각은?",
             "choices":["즉시 대안을 제시한다","상황을 설명하고 기다린다"],"axes":("J","P")}
        ]
    }
    
    # 선택된 장소의 특화 질문이 있으면 추가
    if location in location_specific_questions:
        return base_questions + location_specific_questions[location]
    else:
        return base_questions

# 기본 질문 (호환성을 위해 유지)
core_questions = get_questions_by_location("일반")

# 타이브레이커 질문들
tie_questions = {
    "EI": {"axes":("E","I"), "text":"로봇과 함께하는 활동에서 선호하는 환경은?", "choices":["사람들과 함께하는 분위기","조용하고 집중할 수 있는 공간"]},
    "SN": {"axes":("S","N"), "text":"로봇의 미래 기능에 대한 관심은?", "choices":["현재 실용적인 기능에 집중","미래의 혁신적 가능성에 관심"]},
    "TF": {"axes":("T","F"), "text":"로봇과의 관계에서 중시하는 것은?", "choices":["효율성과 성과","감정적 연결과 이해"]},
    "JP": {"axes":("J","P"), "text":"로봇과의 목표 달성에서 선호하는 방식은?", "choices":["계획적이고 체계적인 접근","유연하고 적응적인 방법"]}
}

# --- MBTI 계산 함수들 ---
def compute_scores(responses):
    scores = {axis: 0 for axis in ['E','I','S','N','T','F','J','P']}
    for q in core_questions:
        choice = responses.get(q['id'])
        if choice is None:
            return None
        pos, neg = q['axes']
        scores[pos if choice == q['choices'][0] else neg] += 1
    return scores

def resolve_ties(scores):
    for axis, cfg in tie_questions.items():
        a, b = cfg['axes']
        if scores[a] == scores[b]:
            tie_choices = ["- 선택하세요 -"] + list(cfg['choices'])
            choice = st.radio(cfg["text"], tie_choices, index=0, key=f"tie_{axis}")
            if choice == "- 선택하세요 -":
                st.warning("추가 설문 문항 응답을 선택해야 진단이 완성됩니다.")
                st.stop()
            scores[a if choice == cfg['choices'][1] else b] += 1
    return scores

def predict_type(scores):
    return ''.join([
        'E' if scores['E'] >= scores['I'] else 'I',
        'S' if scores['S'] >= scores['N'] else 'N',
        'T' if scores['T'] >= scores['F'] else 'F',
        'J' if scores['J'] >= scores['P'] else 'P'
    ])

# --- 데이터 저장 함수들 ---
def save_response(user_id, responses, mbti, scores, profile, robot_id):
    now = datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
    record = {
        "user_id": user_id,
        "gender": profile["gender"],
        "age_group": profile["age_group"],
        "job": profile["job"],
        "robot_id": robot_id,
        "responses": responses,
        "mbti": mbti,
        "scores": scores,
        "timestamp": now
    }
    try:
        supabase.table("responses").insert(record).execute()
        return True
    except Exception as e:
        st.error(f"응답 저장 실패: {e}")
        return False

def load_last_mbti(user_id, robot_id):
    try:
        res = supabase.table("responses").select("*").eq("user_id", user_id).eq("robot_id", robot_id).order("timestamp", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        st.error(f"이전 결과 로드 실패: {e}")
        return None

def generate_adaptive_feedback(curr, prev):
    if prev is None:
        return "이번이 첫 진단 결과입니다. 앞으로 지속적인 자기개선을 기대합니다!"
    elif curr == prev:
        return "지난 번과 동일한 유형입니다. 자신의 강점을 꾸준히 발전시키세요!"
    else:
        return f"이전 유형은 {prev}, 이번엔 {curr}로 변화가 관찰됩니다. 변화를 반영해 HRI 경험을 조정해보세요."

# --- 논문용 통계 분석 함수들 ---
def chi2_mbti_vs_group(df, group_col):
    """MBTI 분포와 집단(성별, 연령대 등) 간의 카이제곱 검정"""
    table = pd.crosstab(df[group_col], df['mbti'])
    chi2, p, dof, expected = chi2_contingency(table)
    return table, chi2, p, dof, expected

def plot_mbti_heatmap(df, group_col):
    """집단별 MBTI 분포 히트맵"""
    table = pd.crosstab(df[group_col], df['mbti'])
    fig = px.imshow(table, title=f"{group_col}별 MBTI 분포 히트맵", aspect="auto")
    return fig, table

def plot_mbti_sankey(df):
    """시간에 따른 MBTI 변화 Sankey Diagram"""
    links = []
    for _, group in df.sort_values('timestamp').groupby('user_id'):
        mbti_list = group['mbti'].tolist()
        for i in range(len(mbti_list)-1):
            links.append((mbti_list[i], mbti_list[i+1]))
    if not links:
        return None, None
    link_df = pd.DataFrame(links, columns=['source','target'])
    link_counts = link_df.value_counts().reset_index(name='count')
    mbti_types = sorted(df['mbti'].unique())
    mbti_idx = {mbti:i for i,mbti in enumerate(mbti_types)}
    fig = go.Figure(go.Sankey(
        node=dict(label=mbti_types),
        link=dict(
            source=[mbti_idx[s] for s in link_counts['source']],
            target=[mbti_idx[t] for t in link_counts['target']],
            value=link_counts['count']
        )
    ))
    fig.update_layout(title_text="MBTI 변화 Sankey Diagram", height=400)
    return fig, link_counts

def plot_corr_heatmap(df):
    """MBTI 유형 간 상관행렬 히트맵"""
    mbti_dummies = pd.get_dummies(df['mbti'])
    corr = mbti_dummies.corr()
    fig = px.imshow(corr, title="MBTI 유형 간 상관행렬", aspect="auto", color_continuous_scale="RdBu")
    return fig, corr

def download_df(df, filetype='csv'):
    """데이터프레임을 CSV/Excel/JSON으로 다운로드"""
    # Excel 시간대 오류 방지를 위해 datetime 컬럼 처리
    df_export = df.copy()
    datetime_columns = df_export.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
    for col in datetime_columns:
        df_export[col] = df_export[col].dt.tz_localize(None) if df_export[col].dt.tz is not None else df_export[col]
    
    if filetype=='csv':
        return df_export.to_csv(index=False).encode('utf-8'), 'data.csv', 'text/csv'
    elif filetype=='excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False)
        return output.getvalue(), 'data.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filetype=='json':
        return df_export.to_json(orient='records', force_ascii=False).encode('utf-8'), 'data.json', 'application/json'
    else:
        return None, '', ''

def generate_report(df, stats_summary, fig_dict):
    """주요 통계/그래프/해석을 Markdown 리포트로 자동 생성"""
    report = """# MBTI HRI 논문용 분석 리포트

## 1. 데이터 요약
- 총 진단 수: {total}
- MBTI 유형 수: {mbti_n}
- 사용자 수: {user_n}

## 2. 주요 통계 결과
""".format(total=len(df), mbti_n=df['mbti'].nunique(), user_n=df['user_id'].nunique())
    for k,v in stats_summary.items():
        report += f"- {k}: {v}\n"
    report += "\n## 3. 주요 Figure 설명\n"
    for k in fig_dict:
        report += f"- {k}\n"
    report += "\n(그래프 이미지는 웹에서 확인/다운로드하세요.)\n"
    return report.encode('utf-8'), 'report.md', 'text/markdown'

# --- 관리자 기능 ---
def check_admin_login(username, password):
    """관리자 로그인 확인"""
    admin_credentials = {
        "admin": "admin123",
        "manager": "manager123"
    }
    return username in admin_credentials and admin_credentials[username] == password

def reset_all_data():
    """전체 데이터 리셋"""
    try:
        # Supabase에서 모든 데이터 삭제
        supabase.table("mbti_responses").delete().neq("id", 0).execute()
        supabase.table("user_robots").delete().neq("id", 0).execute()
        return True, "모든 데이터가 성공적으로 삭제되었습니다."
    except Exception as e:
        return False, f"데이터 삭제 중 오류 발생: {e}"

def get_user_data_summary():
    """사용자 데이터 요약"""
    try:
        users = supabase.table("mbti_responses").select("user_id").execute()
        robots = supabase.table("user_robots").select("*").execute()
        return {
            "total_users": len(set([u['user_id'] for u in users.data])) if users.data else 0,
            "total_responses": len(users.data) if users.data else 0,
            "total_robots": len(robots.data) if robots.data else 0
        }
    except Exception as e:
        return {"error": str(e)}

def delete_user_data(user_id):
    """특정 사용자 데이터 삭제"""
    try:
        supabase.table("mbti_responses").delete().eq("user_id", user_id).execute()
        supabase.table("user_robots").delete().eq("user_id", user_id).execute()
        return True, f"사용자 {user_id}의 데이터가 삭제되었습니다."
    except Exception as e:
        return False, f"사용자 데이터 삭제 중 오류: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="MBTI HRI 통합 분석툴", page_icon="🤖", layout="wide")

# 사이드바 - 사용자/로봇 정보
with st.sidebar:
    st.header("👤 사용자/로봇 정보")
    profile = st.session_state.user_profile
    gender = st.selectbox("성별", ["남", "여"], index=0 if profile["gender"]=="남" else 1)
    age_group_list = ["10대","20대","30대","40대","50대+"]
    age_group = st.selectbox("연령대", age_group_list, index=age_group_list.index(profile["age_group"]))
    job_list = ["학생","연구원","교수","회사원","기타"]
    job_sel_idx = job_list.index(profile["job"]) if profile["job"] in job_list else job_list.index("기타")
    job_sel = st.selectbox("직업", job_list, index=job_sel_idx)
    if job_sel == "기타":
        job_detail = st.text_input("직업 직접 입력", value=profile["job"] if profile["job"] not in job_list else "")
        job_final = job_detail if job_detail.strip() else "기타"
    else:
        job_final = job_sel
    st.session_state.user_profile = {"gender": gender, "age_group": age_group, "job": job_final}

    st.divider()
    
    # 장소 선택 섹션
    st.subheader("🏢 진단 장소 선택")
    location_options = ["일반", "병원", "도서관", "쇼핑몰", "학교", "공항"]
    selected_location = st.selectbox(
        "진단할 장소를 선택하세요",
        location_options,
        index=0,
        help="장소별로 다른 설문 내용이 제공됩니다"
    )
    
    # 선택된 장소에 따른 질문 업데이트
    if 'current_questions' not in st.session_state:
        st.session_state.current_questions = get_questions_by_location(selected_location)
    elif st.session_state.get('selected_location', '일반') != selected_location:
        st.session_state.current_questions = get_questions_by_location(selected_location)
        st.session_state.selected_location = selected_location
    
    # 장소별 설명
    location_descriptions = {
        "일반": "일반적인 HRI 상황에 대한 진단",
        "병원": "의료 환경에서의 로봇 상호작용 진단",
        "도서관": "도서관 환경에서의 로봇 상호작용 진단",
        "쇼핑몰": "쇼핑 환경에서의 로봇 상호작용 진단",
        "학교": "교육 환경에서의 로봇 상호작용 진단",
        "공항": "여행 환경에서의 로봇 상호작용 진단"
    }
    
    st.caption(f"📋 {location_descriptions[selected_location]}")
    st.caption(f"총 {len(st.session_state.current_questions)}개 질문")

    st.divider()
    
    # 관리자 로그인 섹션
    st.subheader("🔧 관리자 로그인")
    
    # 관리자 로그인 상태 확인
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    
    if not st.session_state.admin_logged_in:
        admin_username = st.text_input("관리자 ID", key="admin_username_sidebar")
        admin_password = st.text_input("비밀번호", type="password", key="admin_password_sidebar")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("로그인", key="admin_login_btn"):
                if check_admin_login(admin_username, admin_password):
                    st.session_state.admin_logged_in = True
                    st.success("관리자 로그인 성공!")
                    st.rerun()
                else:
                    st.error("잘못된 로그인 정보입니다.")
        
        with col2:
            if st.button("로그아웃", key="admin_logout_btn"):
                st.session_state.admin_logged_in = False
                st.rerun()
        
        st.caption("관리자 계정: admin/admin123 또는 manager/manager123")
    else:
        st.success("관리자 로그인됨")
        if st.button("로그아웃", key="admin_logout_btn2"):
            st.session_state.admin_logged_in = False
            st.rerun()
    
    st.divider()
    st.subheader("🤖 로봇 ID 관리")
    
    # 데이터베이스 상태 확인
    try:
        supabase.table("user_robots").select("id").limit(1).execute()
        db_status = "✅ 데이터베이스 연결됨"
    except Exception as e:
        if "does not exist" in str(e):
            db_status = "⚠️ 테이블 미생성 (로컬 모드)"
        else:
            db_status = "❌ 데이터베이스 오류"
    
    st.caption(f"상태: {db_status}")
    
    # 로봇 목록 불러오기
    robot_opts = list(st.session_state.robot_list)
    
    # 새 로봇 등록
    col1, col2 = st.columns([3, 1])
    with col1:
        new_robot = st.text_input("새 로봇 별칭 등록(예: 내식기1)", key="new_robot_id")
    with col2:
        if st.button("➕ 등록"):
            if new_robot.strip() and new_robot.strip() not in robot_opts:
                # Supabase에 저장
                if save_robot_to_db(USER_ID, new_robot.strip()):
                    robot_opts.insert(0, new_robot.strip())
                    st.session_state.robot_list = robot_opts
                    st.success(f"✅ '{new_robot.strip()}' 등록 완료!")
                    st.rerun()
                else:
                    st.error("로봇 등록에 실패했습니다.")
            elif new_robot.strip() in robot_opts:
                st.warning("이미 등록된 로봇입니다.")
            else:
                st.warning("로봇 이름을 입력해주세요.")
    
    # 로봇 선택
    if robot_opts:
        robot_id = st.selectbox("진단 대상 로봇(선택)", robot_opts, 
                               index=0 if st.session_state.robot_id not in robot_opts else robot_opts.index(st.session_state.robot_id))
        st.session_state.robot_id = robot_id
        
        # 로봇 삭제 기능
        if len(robot_opts) > 1:  # 최소 1개는 남겨두기
            st.subheader("🗑️ 로봇 삭제")
            delete_robot = st.selectbox("삭제할 로봇 선택", robot_opts, key="delete_robot")
            if st.button("🗑️ 삭제"):
                if delete_robot_from_db(USER_ID, delete_robot):
                    robot_opts.remove(delete_robot)
                    st.session_state.robot_list = robot_opts
                    if st.session_state.robot_id == delete_robot:
                        st.session_state.robot_id = robot_opts[0]
                    st.success(f"✅ '{delete_robot}' 삭제 완료!")
                    st.rerun()
                else:
                    st.error("로봇 삭제에 실패했습니다.")
        else:
            st.info("로봇은 최소 1개 이상 유지해야 합니다.")
    else:
        st.warning("등록된 로봇이 없습니다. 새 로봇을 등록해주세요.")
    
    # 관리자 기능 (로그인된 경우에만 표시)
    if st.session_state.admin_logged_in:
        st.divider()
        st.subheader("🗑️ 관리자 기능")
        
        # 데이터 요약
        summary = get_user_data_summary()
        if "error" not in summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 사용자", summary["total_users"])
            with col2:
                st.metric("총 응답", summary["total_responses"])
            with col3:
                st.metric("총 로봇", summary["total_robots"])
        
        # 데이터 관리
        if st.button("전체 데이터 리셋", type="primary", key="reset_all_sidebar"):
            if st.button("정말 삭제하시겠습니까?", key="confirm_reset_sidebar"):
                success, message = reset_all_data()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

if 'page' not in st.session_state: st.session_state.page = 1
page = st.session_state.page

######### 1. 진단 #############
if page == 1:
    # 진단(설문) 시작할 때 Flag를 False로 리셋!
    st.session_state['saved_result'] = False

    st.header("1️⃣ MBTI 기반 HRI UX 진단")
    
    consent = st.checkbox("익명 데이터 분석 활용에 동의합니다.", value=True)
    if not consent:
        st.warning("진단 시작엔 동의가 필요합니다!")
        st.stop()
    
    responses = {}
    current_questions = st.session_state.get('current_questions', core_questions)
    total_questions = len(current_questions)
    
    # 선택된 장소 표시
    selected_location = st.session_state.get('selected_location', '일반')
    st.info(f"🏢 진단 장소: **{selected_location}** - 총 {total_questions}개 질문")
    
    # 2개씩 질문을 표시
    for i in range(0, len(current_questions), 2):
        col1, col2 = st.columns(2)
        
        # 첫 번째 질문
        with col1:
            q1 = current_questions[i]
            st.write(f"**{i + 1}. {q1['text']}**")
            selected1 = st.radio(
                "선택해주세요:",
                q1['choices'],
                key=f"radio_{q1['id']}",
                label_visibility="collapsed"
            )
            if selected1:
                responses[q1['id']] = selected1
        
        # 두 번째 질문 (있는 경우)
        with col2:
            if i + 1 < len(current_questions):
                q2 = current_questions[i + 1]
                st.write(f"**{i + 2}. {q2['text']}**")
                selected2 = st.radio(
                    "선택해주세요:",
                    q2['choices'],
                    key=f"radio_{q2['id']}",
                    label_visibility="collapsed"
                )
                if selected2:
                    responses[q2['id']] = selected2
        
        st.divider()
    
    st.session_state['responses'] = responses
    
    # 결과 보기 버튼
    if st.button("🎯 결과 보기", type="primary", use_container_width=True):
        if len(responses) < total_questions:
            st.warning("모든 문항에 답변해주세요!")
        else:
            st.session_state.page = 2
            st.rerun()

######### 2. 결과 #############
elif page == 2:
    st.header(f"2️⃣ [{st.session_state.robot_id}] 진단 결과·피드백")
    
    # 로딩 애니메이션
    with st.spinner("🔍 MBTI 분석 중..."):
        time.sleep(1)
    
    responses = st.session_state.get('responses', {})
    profile = st.session_state.user_profile
    robot_id = st.session_state.robot_id
    scores = compute_scores(responses)
    
    if scores is None:
        st.warning("모든 문항에 답해주세요.")
    else:
        scores = resolve_ties(scores)
        mbti = predict_type(scores)
        
        if not st.session_state.get('saved_result', False):
            # 장소 정보도 함께 저장
            profile_with_location = profile.copy()
            profile_with_location['location'] = selected_location
            save_response(USER_ID, responses, mbti, scores, profile_with_location, robot_id)
            st.session_state['saved_result'] = True   # 저장함 표시

        prev_record = load_last_mbti(USER_ID, robot_id)
        prev_mbti = prev_record['mbti'] if prev_record else None
        
        # 결과 표시 개선
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ## 🎯 진단 결과
            
            ### 🤖 로봇: **{robot_id}**
            ### 🏢 장소: **{selected_location}**
            ### 🧠 MBTI 유형: **{mbti}**
            """)
            
            # MBTI 유형별 색상 표시
            mbti_color = mbti_colors.get(mbti, '#CCCCCC')
            st.markdown(f"""
            <div style="background-color: {mbti_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2>🎨 {mbti}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 점수 분포 시각화
            st.subheader("📊 점수 분포")
            score_data = {
                '축': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'],
                '점수': [scores['E'], scores['I'], scores['S'], scores['N'], 
                        scores['T'], scores['F'], scores['J'], scores['P']]
            }
            score_df = pd.DataFrame(score_data)
            
            fig = px.bar(score_df, x='축', y='점수', 
                        title="MBTI 축별 점수",
                        color='축',
                        color_discrete_map=mbti_colors)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 적응적 피드백
        st.subheader("💡 개인화된 피드백")
        feedback = generate_adaptive_feedback(mbti, prev_mbti)
        st.info(feedback)
        
        # MBTI 가이드
        guide = guide_data.get(mbti)
        if guide:
            st.subheader("📖 MBTI 유형 가이드")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**🎭 성격 특징:**")
                st.write(guide['description'])
            
            with col2:
                st.write("**🤖 HRI 상호작용 스타일:**")
                st.info(guide.get('hri_style', 'HRI 스타일 정보가 없습니다.'))
                with st.expander("📋 시나리오 예시"):
                    for i, ex in enumerate(guide['examples'], 1):
                        st.write(f"{i}. {ex}")
        
        # 이전 결과와 비교
        if prev_mbti:
            st.subheader("📈 변화 추이")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("이전 MBTI", prev_mbti)
            with col2:
                st.metric("현재 MBTI", mbti)
            
            if prev_mbti != mbti:
                st.success("🎉 MBTI 유형이 변화했습니다!")
            else:
                st.info("📊 MBTI 유형이 일관성을 보입니다.")
        
        # 다운로드 및 다음 단계
        st.subheader("💾 결과 저장")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 결과 CSV 다운로드", 
                             pd.DataFrame([{'유형': mbti, '로봇': robot_id, '날짜': datetime.now().strftime('%Y-%m-%d')}]).to_csv(index=False), 
                             f"{mbti}_{robot_id}.csv")
        with col2:
            if st.button("📊 통계/히스토리 대시보드 이동", type="primary", use_container_width=True):
                st.session_state.page = 3
                st.rerun()

######### 3. 통계/히스토리 #############
elif page == 3:
    st.header("3️⃣ 전체 통계 · 로봇 이력 · 집단분석(통합)")
    
    # 탭으로 구분하여 더 깔끔한 UI 제공
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 전체 트렌드", "📈 집단별 분석", "🤖 로봇 이력", 
        "🧠 고급 분석", "🎯 개인 대시보드", "🤖 AI 인사이트", 
        "📋 데이터 관리", "📈 논문용 분석", "🔧 관리자"
    ])
    
    try:
        res = supabase.table("responses").select("*").execute()
        if not res.data or len(res.data)==0:
            st.info("아직 데이터가 없습니다.")
        else:
            df = pd.DataFrame(res.data)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            with tab1:
                st.subheader("📊 기간별 MBTI 트렌드")
                min_date, max_date = df['date'].min(), df['date'].max()
                
                # chart_type 변수를 먼저 정의
                chart_type = "라인"  # 기본값 설정
                
                if min_date == max_date:
                    st.info(f"데이터 날짜: {min_date}")
                    df_period = df[df['date']==min_date]
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        date_sel = st.slider("조회 기간", min_value=min_date, max_value=max_date, 
                                           value=(min_date, max_date), format="YYYY-MM-DD")
                    with col2:
                        chart_type = st.selectbox("차트 유형", ["라인", "바", "영역"])
                    
                    df_period = df[(df['date']>=date_sel[0])&(df['date']<=date_sel[1])]
                
                if not df_period.empty:
                    # Plotly를 사용한 인터랙티블 차트
                    daily_mbti = df_period.groupby(['date', 'mbti']).size().reset_index(name='count')
                    
                    if chart_type == "라인":
                        fig = px.line(daily_mbti, x='date', y='count', color='mbti',
                                    title="기간별 MBTI 트렌드", color_discrete_map=mbti_colors)
                    elif chart_type == "바":
                        fig = px.bar(daily_mbti, x='date', y='count', color='mbti',
                                   title="기간별 MBTI 분포", color_discrete_map=mbti_colors)
                    else:  # 영역
                        fig = px.area(daily_mbti, x='date', y='count', color='mbti',
                                    title="기간별 MBTI 누적 분포", color_discrete_map=mbti_colors)
                    
                    fig.update_layout(height=500, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 요약 통계
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 진단 수", len(df_period))
                    with col2:
                        st.metric("MBTI 유형 수", df_period['mbti'].nunique())
                    with col3:
                        st.metric("가장 많은 유형", df_period['mbti'].mode().iloc[0] if not df_period['mbti'].mode().empty else "N/A")
            
            with tab2:
                st.subheader("📈 집단별 MBTI 분포 분석")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    group_col = st.selectbox("분포 분석 기준", ["gender", "age_group", "job", "robot_id"])
                    chart_style = st.selectbox("차트 스타일", ["바 차트", "파이 차트", "히트맵"])
                
                with col2:
                    group_df = df.groupby([group_col, "mbti"]).size().unstack(fill_value=0)
                    
                    if chart_style == "바 차트":
                        fig = px.bar(group_df, title=f"{group_col}별 MBTI 분포", 
                                   color_discrete_map=mbti_colors)
                        fig.update_layout(height=400, xaxis_title=group_col, yaxis_title="진단 수")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_style == "파이 차트":
                        # 각 그룹별 파이 차트를 서브플롯으로 표시
                        categories = group_df.index
                        n_cats = len(categories)
                        cols = min(3, n_cats)
                        rows = (n_cats + cols - 1) // cols
                        
                        fig = make_subplots(rows=rows, cols=cols, 
                                          specs=[[{"type": "pie"}] * cols] * rows,
                                          subplot_titles=categories)
                        
                        for i, cat in enumerate(categories):
                            row = i // cols + 1
                            col = i % cols + 1
                            
                            values = group_df.loc[cat].values
                            labels = group_df.columns
                            
                            fig.add_trace(
                                go.Pie(labels=labels, values=values, name=cat,
                                      marker_colors=[mbti_colors.get(mbti, '#CCCCCC') for mbti in labels]),
                                row=row, col=col
                            )
                        
                        fig.update_layout(height=300 * rows, title_text=f"{group_col}별 MBTI 분포")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # 히트맵
                        fig = px.imshow(group_df, title=f"{group_col}별 MBTI 히트맵",
                                      aspect="auto", color_continuous_scale="Viridis")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # 피벗테이블
                st.subheader("📋 상세 데이터 테이블")
                pivot1 = pd.pivot_table(df, index=group_col, columns="mbti", aggfunc="size", fill_value=0)
                pivot2 = pd.pivot_table(df, index="mbti", columns=group_col, aggfunc="size", fill_value=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**세로: 집단, 가로: MBTI**")
                    st.dataframe(pivot1, use_container_width=True)
                with col2:
                    st.write("**세로: MBTI, 가로: 집단**")
                    st.dataframe(pivot2, use_container_width=True)
            
            with tab3:
                st.subheader(f"🤖 '{st.session_state.robot_id}'의 MBTI 변화 히스토리")
                
                bot_records = df[(df['user_id']==USER_ID) & (df['robot_id']==st.session_state.robot_id)].sort_values("timestamp")
                
                if not bot_records.empty:
                    # 인터랙티블 타임라인 차트
                    bot_records['timestamp_short'] = pd.to_datetime(bot_records['timestamp']).dt.strftime("%Y-%m-%d")
                    
                    fig = px.scatter(bot_records, x='timestamp_short', y='mbti', 
                                   title=f"'{st.session_state.robot_id}' MBTI 변화 타임라인",
                                   color='mbti', color_discrete_map=mbti_colors,
                                   size=[20] * len(bot_records))  # 모든 점을 동일한 크기로
                    
                    fig.update_layout(height=400, xaxis_title="날짜", yaxis_title="MBTI 유형")
                    fig.update_traces(marker=dict(size=15))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 변화 분석
                    if len(bot_records) > 1:
                        # MBTI 변화를 문자열로 계산
                        mbti_list = bot_records['mbti'].tolist()
                        changes = []
                        for i in range(1, len(mbti_list)):
                            if mbti_list[i] != mbti_list[i-1]:
                                changes.append(f"{mbti_list[i-1]} → {mbti_list[i]}")
                        
                        if changes:
                            st.info(f"총 {len(changes)}번의 변화가 있었습니다.")
                            
                            # 변화 패턴 분석
                            change_counts = pd.Series(changes).value_counts()
                            if not change_counts.empty:
                                st.write("**주요 변화 패턴:**")
                                for change, count in change_counts.head(3).items():
                                    st.write(f"- {change}: {count}회")
                        else:
                            st.info("변화가 없었습니다.")
                    
                    # 상세 데이터
                    st.dataframe(bot_records[["timestamp", "mbti", "gender", "age_group", "job"]], 
                               use_container_width=True)
                else:
                    st.info(f"로봇 '{st.session_state.robot_id}'의 진단 이력이 없습니다.")
            
            with tab4:
                st.subheader("🧠 고급 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 MBTI 상관관계 분석")
                    if len(df) > 1:
                        correlations = calculate_mbti_correlations(df)
                        if correlations is not None:
                            fig = px.imshow(correlations, 
                                          title="MBTI 유형 간 상관관계",
                                          color_continuous_scale="RdBu",
                                          aspect="auto")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("상관관계 분석을 위한 충분한 데이터가 없습니다.")
                    
                    st.subheader("🎯 MBTI 트렌드 예측")
                    if len(df) > 0:
                        predicted_mbti, prediction_msg = predict_mbti_trends(df, USER_ID, st.session_state.robot_id)
                        if predicted_mbti:
                            st.success(f"🔮 {prediction_msg}")
                        else:
                            st.info(prediction_msg)
                
                with col2:
                    st.subheader("🌐 MBTI 네트워크 분석")
                    if len(df) > 1:
                        network, network_msg = create_mbti_network(df)
                        if network:
                            st.success(network_msg)
                            
                            # 네트워크 시각화
                            pos = nx.spring_layout(network, k=1, iterations=50)
                            
                            # 노드와 엣지 데이터 준비
                            node_x = []
                            node_y = []
                            node_text = []
                            node_size = []
                            
                            for node in network.nodes():
                                x, y = pos[node]
                                node_x.append(x)
                                node_y.append(y)
                                node_text.append(node)
                                node_size.append(len([n for n in network.neighbors(node)]) * 10 + 10)
                            
                            edge_x = []
                            edge_y = []
                            edge_weights = []
                            
                            for edge in network.edges(data=True):
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_x.extend([x0, x1, None])
                                edge_y.extend([y0, y1, None])
                                edge_weights.append(edge[2]['weight'])
                            
                            # 엣지 트레이스
                            edge_trace = go.Scatter(
                                x=edge_x, y=edge_y,
                                line=dict(width=2, color='#888'),
                                hoverinfo='none',
                                mode='lines')
                            
                            # 노드 트레이스
                            node_trace = go.Scatter(
                                x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                text=node_text,
                                textposition="middle center",
                                marker=dict(
                                    size=node_size,
                                    color=[mbti_colors.get(mbti, '#CCCCCC') for mbti in node_text],
                                    line=dict(width=2, color='white')
                                ))
                            
                            fig = go.Figure(data=[edge_trace, node_trace],
                                          layout=go.Layout(
                                              title="MBTI 유형 간 관계 네트워크",
                                              showlegend=False,
                                              hovermode='closest',
                                              margin=dict(b=20,l=5,r=5,t=40),
                                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                          )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(network_msg)
                    
                    st.subheader("🔍 사용자 클러스터링")
                    if len(df) > 2:
                        clustered_df, cluster_msg = cluster_users_by_mbti(df)
                        if clustered_df is not None:
                            st.success(cluster_msg)
                            
                            # 클러스터별 분포 시각화
                            cluster_counts = clustered_df.groupby('cluster').size()
                            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                        title="클러스터별 사용자 분포",
                                        labels={'x': '클러스터', 'y': '사용자 수'})
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(cluster_msg)
            
            with tab5:
                st.subheader("🎯 개인 대시보드")
                
                # 개인 진화 타임라인
                st.subheader("📈 나의 MBTI 진화 타임라인")
                my_all_records = df[df['user_id']==USER_ID].sort_values('timestamp')
                
                if not my_all_records.empty:
                    # 인터랙티블 타임라인
                    my_all_records['timestamp_short'] = pd.to_datetime(my_all_records['timestamp']).dt.strftime("%Y-%m-%d")
                    
                    fig = px.scatter(my_all_records, x='timestamp_short', y='mbti', 
                                   color='robot_id', size=[20] * len(my_all_records),
                                   title="나의 모든 로봇 MBTI 변화",
                                   color_discrete_map=mbti_colors)
                    
                    fig.update_layout(height=400, xaxis_title="날짜", yaxis_title="MBTI 유형")
                    fig.update_traces(marker=dict(size=15))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 개인 통계
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 진단 수", len(my_all_records))
                    with col2:
                        st.metric("사용한 로봇 수", my_all_records['robot_id'].nunique())
                    with col3:
                        st.metric("MBTI 유형 수", my_all_records['mbti'].nunique())
                    with col4:
                        most_used_robot = my_all_records['robot_id'].mode().iloc[0] if not my_all_records['robot_id'].mode().empty else "N/A"
                        st.metric("가장 많이 사용한 로봇", most_used_robot)
                    
                    # MBTI 변화 히트맵
                    st.subheader("🔥 MBTI 변화 히트맵")
                    mbti_pivot = my_all_records.pivot_table(index='robot_id', columns='mbti', aggfunc='size', fill_value=0)
                    
                    fig = px.imshow(mbti_pivot, 
                                  title="로봇별 MBTI 분포 히트맵",
                                  color_continuous_scale="Viridis",
                                  aspect="auto")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("아직 진단 이력이 없습니다. 첫 진단을 시작해보세요!")
            
            with tab6:
                st.subheader("🤖 AI 인사이트")
                
                # AI 인사이트 생성
                insights = generate_ai_insights(df, USER_ID, st.session_state.robot_id)
                
                if insights:
                    st.success("✨ AI가 분석한 인사이트")
                    
                    for insight in insights:
                        st.write(f"• {insight}")
                    
                    # 인사이트 시각화
                    st.subheader("📊 인사이트 시각화")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 연령대별 MBTI 분포
                        if 'age_group' in df.columns and len(df) > 0:
                            age_mbti = df.groupby(['age_group', 'mbti']).size().unstack(fill_value=0)
                            
                            fig = px.bar(age_mbti, title="연령대별 MBTI 분포",
                                       color_discrete_map=mbti_colors)
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 성별 MBTI 분포
                        if 'gender' in df.columns and len(df) > 0:
                            gender_mbti = df.groupby(['gender', 'mbti']).size().unstack(fill_value=0)
                            
                            fig = px.bar(gender_mbti, title="성별 MBTI 분포",
                                       color_discrete_map=mbti_colors)
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D 차트
                    st.subheader("🌍 3D MBTI 분포 분석")
                    fig_3d, msg_3d = create_3d_mbti_chart(df)
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info(msg_3d)
                else:
                    st.info("AI 인사이트를 생성할 수 있는 충분한 데이터가 없습니다.")
            
            with tab7:
                st.subheader("📈 진행률 추적")
            
            with tab7:
                st.subheader("📋 데이터 관리")
                
                # 내 모든 로봇 MBTI 이력
                with st.expander("내 모든 로봇 MBTI 진단/변화 이력", expanded=True):
                    my_records = df[df['user_id']==USER_ID].sort_values('timestamp')
                    if not my_records.empty:
                        # 인터랙티블 테이블
                        st.dataframe(my_records[["timestamp", "robot_id", "mbti", "gender", "age_group", "job"]], 
                                   use_container_width=True)
                        
                        # 요약 통계
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 진단 수", len(my_records))
                        with col2:
                            st.metric("사용한 로봇 수", my_records['robot_id'].nunique())
                        with col3:
                            st.metric("MBTI 유형 수", my_records['mbti'].nunique())
                    else:
                        st.info("아직 진단 이력이 없습니다.")
                
                # 데이터 다운로드
                st.subheader("📥 데이터 다운로드")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("전체 데이터 CSV", 
                                     df.to_csv(index=False).encode("utf-8"), 
                                     "all_responses.csv", "text/csv")
                with col2:
                    st.download_button("전체 데이터 JSON", 
                                     df.to_json(orient="records", force_ascii=False).encode("utf-8"), 
                                     "all_responses.json", "application/json")
                
                # HRI 모델 정보
                if hri_models:
                    st.subheader("🤖 HRI 모델(시나리오) 목록")
                    for m in hri_models:
                        st.markdown(f"**{m.get('name','모델명 없음')}**: {m.get('description','')}")
            
            with tab8:
                st.header("📈 논문용 고급 분석 및 리포트")
                try:
                    # 데이터 준비
                    if not df.empty:
                        # 1. 카이제곱 검정 (집단별 MBTI 분포)
                        st.subheader("카이제곱 검정: 집단별 MBTI 분포")
                        group_col = st.selectbox("집단 선택", ['gender','age_group','job','robot_id'], key="paper_group")
                        table = pd.crosstab(df[group_col], df['mbti'])
                        from scipy.stats import chi2_contingency
                        chi2, p, dof, expected = chi2_contingency(table)
                        st.write(f"카이제곱 통계량: {chi2:.2f}, 자유도: {dof}, p-value: {p:.4f}")
                        if p < 0.05:
                            st.success("집단별 MBTI 분포 차이가 통계적으로 유의미합니다 (p<0.05)")
                        else:
                            st.info("집단별 MBTI 분포 차이가 통계적으로 유의하지 않습니다 (p>=0.05)")
                        st.write(table)
                        # 2. Sankey Diagram (MBTI 변화)
                        st.subheader("MBTI 변화 Sankey Diagram")
                        links = []
                        for _, group in df.sort_values('timestamp').groupby('user_id'):
                            mbti_list = group['mbti'].tolist()
                            for i in range(len(mbti_list)-1):
                                links.append((mbti_list[i], mbti_list[i+1]))
                        if links:
                            link_df = pd.DataFrame(links, columns=['source','target'])
                            link_counts = link_df.value_counts().reset_index(name='count')
                            mbti_types = sorted(df['mbti'].unique())
                            mbti_idx = {mbti:i for i,mbti in enumerate(mbti_types)}
                            import plotly.graph_objects as go
                            fig = go.Figure(go.Sankey(
                                node=dict(label=mbti_types),
                                link=dict(
                                    source=[mbti_idx[s] for s in link_counts['source']],
                                    target=[mbti_idx[t] for t in link_counts['target']],
                                    value=link_counts['count']
                                )
                            ))
                            fig.update_layout(title_text="MBTI 변화 Sankey Diagram", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Sankey 그래프를 그릴 데이터가 부족합니다.")
                        # 3. 상관행렬 히트맵
                        st.subheader("MBTI 유형 간 상관행렬 히트맵")
                        mbti_dummies = pd.get_dummies(df['mbti'])
                        corr = mbti_dummies.corr()
                        import plotly.express as px
                        fig_corr = px.imshow(corr, title="MBTI 유형 간 상관행렬", aspect="auto", color_continuous_scale="RdBu")
                        st.plotly_chart(fig_corr, use_container_width=True)
                        

                        # 5. 자동 리포트 생성 및 다운로드
                        st.subheader("📝 분석 리포트 다운로드")
                        stats_summary = {
                            '카이제곱 p-value': f"{p:.4f}",
                            '상관행렬 평균': f"{corr.values.mean():.3f}",
                            'MBTI 변화쌍 수': len(links)
                        }
                        report = f"""# MBTI HRI 논문용 분석 리포트\n\n## 1. 데이터 요약\n- 총 진단 수: {len(df)}\n- MBTI 유형 수: {df['mbti'].nunique()}\n- 사용자 수: {df['user_id'].nunique()}\n\n## 2. 주요 통계 결과\n"""
                        for k,v in stats_summary.items():
                            report += f"- {k}: {v}\n"
                        report += "\n## 3. 분석 차트\n- 카이제곱 검정: 집단별 MBTI 분포 차이 분석\n- Sankey Diagram: MBTI 변화 패턴 시각화\n- 상관행렬 히트맵: MBTI 유형 간 상관관계\n\n(그래프 이미지는 웹에서 확인/다운로드하세요.)\n"
                        st.download_button("리포트(Markdown) 다운로드", report.encode('utf-8'), "report.md", "text/markdown")
                        # 6. 데이터 다운로드
                        st.subheader("📥 데이터 다운로드")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button("CSV 다운로드", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
                        with col2:
                            import io
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False)
                            st.download_button("Excel 다운로드", output.getvalue(), "data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        with col3:
                            st.download_button("JSON 다운로드", df.to_json(orient='records', force_ascii=False).encode('utf-8'), "data.json", "application/json")
                        st.write("---")
                        st.write("**데이터 미리보기:**")
                        st.dataframe(df.head())
                    else:
                        st.info("분석할 데이터가 없습니다.")
                except Exception as e:
                    st.error(f"논문용 분석 오류: {e}")
            
            with tab9:
                st.header("🔧 관리자 전용 기능")
                
                # 관리자 로그인
                if 'admin_logged_in' not in st.session_state:
                    st.session_state.admin_logged_in = False
                
                if not st.session_state.admin_logged_in:
                    st.subheader("관리자 로그인")
                    col1, col2 = st.columns(2)
                    with col1:
                        admin_username = st.text_input("관리자 ID", key="admin_username")
                    with col2:
                        admin_password = st.text_input("비밀번호", type="password", key="admin_password")
                    
                    if st.button("로그인"):
                        if check_admin_login(admin_username, admin_password):
                            st.session_state.admin_logged_in = True
                            st.success("관리자 로그인 성공!")
                            st.rerun()
                        else:
                            st.error("잘못된 로그인 정보입니다.")
                    
                    st.info("관리자 계정: admin/admin123 또는 manager/manager123")
                
                else:
                    st.success("관리자 로그인됨")
                    
                    # 데이터 요약
                    st.subheader("📊 데이터 요약")
                    summary = get_user_data_summary()
                    if "error" not in summary:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 사용자 수", summary["total_users"])
                        with col2:
                            st.metric("총 응답 수", summary["total_responses"])
                        with col3:
                            st.metric("총 로봇 수", summary["total_robots"])
                    else:
                        st.error(f"데이터 요약 오류: {summary['error']}")
                    
                    st.divider()
                    
                    # 데이터 관리
                    st.subheader("🗑️ 데이터 관리")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("전체 데이터 리셋", type="primary"):
                            if st.button("정말 삭제하시겠습니까?", key="confirm_reset"):
                                success, message = reset_all_data()
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    with col2:
                        if st.button("로그아웃"):
                            st.session_state.admin_logged_in = False
                            st.rerun()
                    
                    # 사용자별 데이터 관리
                    st.subheader("👥 사용자별 데이터 관리")
                    try:
                        users_data = supabase.table("mbti_responses").select("user_id").execute()
                        if users_data.data:
                            user_list = list(set([u['user_id'] for u in users_data.data]))
                            selected_user = st.selectbox("사용자 선택", user_list)
                            
                            if st.button(f"사용자 {selected_user} 데이터 삭제"):
                                success, message = delete_user_data(selected_user)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        else:
                            st.info("삭제할 사용자 데이터가 없습니다.")
                    except Exception as e:
                        st.error(f"사용자 데이터 조회 오류: {e}")
    
    except Exception as e:
        st.error(f"통계 대시보드 오류: {e}")
        st.exception(e)

    if st.button("진단 첫화면으로 돌아가기"):
        st.session_state.page = 1
        st.rerun()
st.markdown("---")
st.info("MBTI 기반 로봇 성격 진단툴(로봇ID 직접입력, 그룹/로봇별 분석, 고도화 시각화)")