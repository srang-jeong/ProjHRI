import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import time
import io
import os
from dotenv import load_dotenv
from supabase import create_client
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="로봇 MBTI 진단 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MBTI 색상 매핑
MBTI_COLORS = {
    'ENFJ': '#FF6B6B', 'ENTJ': '#4ECDC4', 'ENTP': '#45B7D1', 'ENFP': '#96CEB4',
    'ESFJ': '#FFEAA7', 'ESFP': '#DDA0DD', 'ESTJ': '#98D8C8', 'ESTP': '#F7DC6F',
    'INFJ': '#BB8FCE', 'INFP': '#85C1E9', 'INTJ': '#F8C471', 'INTP': '#82E0AA',
    'ISFJ': '#F1948A', 'ISFP': '#85C1E9', 'ISTJ': '#F7DC6F', 'ISTP': '#D7BDE2'
}

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# 세션 상태 초기화
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'user_id': None,
        'robot_id': "로봇A",
        'robot_list': ["로봇A"],
        'admin_logged_in': False,
        'selected_location': None,
        'page': 1,
        'saved_result': False,
        'current_diagnosis_id': None,  # 현재 진단 세션 ID 추가
        'user_profile': {"gender": "남", "age_group": "20대", "job": "학생"},
        'registered_users': {
            "admin": "admin123",
            "manager": "manager123"
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_diagnosis_session():
    """진단 세션 상태 초기화"""
    st.session_state.saved_result = False
    st.session_state.current_diagnosis_id = None
    st.session_state.responses = {}

# CSS 스타일 설정
def setup_styles():
    """CSS 스타일 설정"""
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
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
    
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# 사용자 ID 검증 함수들
def validate_user_id(user_id):
    """사용자 ID 유효성 검증"""
    if not user_id or not user_id.strip():
        return False, "사용자 ID를 입력해주세요."
    
    if len(user_id.strip()) < 2:
        return False, "사용자 ID는 2자 이상이어야 합니다."
    
    if len(user_id.strip()) > 20:
        return False, "사용자 ID는 20자 이하여야 합니다."
    
    # 특수문자 제한 (보안성 향상)
    import re
    if not re.match(r'^[a-zA-Z0-9가-힣\s_-]+$', user_id.strip()):
        return False, "사용자 ID는 영문, 숫자, 한글, 공백, 언더스코어(_), 하이픈(-)만 사용 가능합니다."
    
    return True, "유효한 사용자 ID입니다."

def validate_robot_id(robot_id):
    """로봇 ID 유효성 검증"""
    if not robot_id or not robot_id.strip():
        return False, "로봇 ID를 입력해주세요."
    
    if len(robot_id.strip()) < 2:
        return False, "로봇 ID는 2자 이상이어야 합니다."
    
    if len(robot_id.strip()) > 20:
        return False, "로봇 ID는 20자 이하여야 합니다."
    
    # 특수문자 제한
    import re
    if not re.match(r'^[a-zA-Z0-9가-힣\s_-]+$', robot_id.strip()):
        return False, "로봇 ID는 영문, 숫자, 한글, 공백, 언더스코어(_), 하이픈(-)만 사용 가능합니다."
    
    return True, "유효한 로봇 ID입니다."

def sanitize_input(text):
    """입력값 정제 (XSS 방지)"""
    if not text:
        return ""
    
    # HTML 태그 제거
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # 위험한 문자 제거
    text = text.replace('"', '').replace("'", '').replace(';', '').replace('--', '')
    
    return text.strip()

def check_admin_login(username, password):
    """관리자 로그인 확인"""
    admin_credentials = {
        "admin": "admin123",
        "manager": "manager123"
    }
    return username in admin_credentials and admin_credentials[username] == password

# 데이터 관리 함수들
def save_response(user_id, responses, mbti, scores, profile, robot_id):
    """응답 데이터 저장 (보안 강화)"""
    try:
        # 입력값 정제
        user_id = sanitize_input(user_id)
        robot_id = sanitize_input(robot_id)
        
        # 유효성 검증
        user_valid, user_msg = validate_user_id(user_id)
        robot_valid, robot_msg = validate_robot_id(robot_id)
        
        if not user_valid:
            st.error(f"사용자 ID 오류: {user_msg}")
            return False
        
        if not robot_valid:
            st.error(f"로봇 ID 오류: {robot_msg}")
            return False
        
        record = {
            "user_id": user_id,
            "gender": profile["gender"],
            "age_group": profile["age_group"],
            "job": profile["job"],
            "robot_id": robot_id,
            "responses": responses,
            "mbti": mbti,
            "scores": scores,
            "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
        }
        supabase.table("responses").insert(record).execute()
        return True
    except Exception as e:
        st.error(f"응답 저장 실패: {e}")
        return False

def load_responses():
    """모든 응답 데이터 로드"""
    try:
        res = supabase.table("responses").select("*").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

def load_user_robots(user_id):
    """사용자의 로봇 목록 로드"""
    try:
        res = supabase.table("user_robots").select("*").eq("user_id", user_id).execute()
        return [robot['robot_name'] for robot in res.data] if res.data else []
    except Exception as e:
        return []

def save_robot(user_id, robot_name, robot_description=""):
    """로봇 정보 저장 (보안 강화)"""
    try:
        # 입력값 정제
        user_id = sanitize_input(user_id)
        robot_name = sanitize_input(robot_name)
        robot_description = sanitize_input(robot_description)
        
        # 유효성 검증
        user_valid, user_msg = validate_user_id(user_id)
        robot_valid, robot_msg = validate_robot_id(robot_name)
        
        if not user_valid:
            st.error(f"사용자 ID 오류: {user_msg}")
            return False
        
        if not robot_valid:
            st.error(f"로봇 ID 오류: {robot_msg}")
            return False
        
        # 중복 로봇 확인
        try:
            existing = supabase.table("user_robots").select("robot_name").eq("user_id", user_id).eq("robot_name", robot_name).execute()
            if existing.data:
                st.warning(f"이미 등록된 로봇입니다: {robot_name}")
                return False
        except Exception as e:
            st.info(f"중복 확인 중 오류 (무시됨): {e}")
        
        record = {
            "user_id": user_id,
            "robot_name": robot_name,
            "robot_description": robot_description,
            "created_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
        }
        
        # 데이터베이스 저장
        result = supabase.table("user_robots").insert(record).execute()
        
        if result.data:
            return True
        else:
            st.error("데이터베이스 저장 실패: 응답 데이터가 없습니다.")
            return False
            
    except Exception as e:
        st.error(f"로봇 등록 실패: {str(e)}")
        st.info("가능한 원인: 데이터베이스 연결 문제, 테이블 구조 오류, 권한 문제")
        return False

# MBTI 계산 함수들
def load_questions(location="일반"):
    """장소별 특화된 질문 데이터 로드"""
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
    
    # 장소별 특화 질문
    location_specific_questions = {
        "병원": [
            {"id":"H1","text":"병원에서 로봇이 환자 정보를 확인할 때 당신의 반응은?",
             "choices":["즉시 필요한 정보를 제공한다","먼저 로봇의 신뢰성을 확인한다"],"axes":("E","I")},
            {"id":"H2","text":"로봇이 의료진과 함께 있을 때 선호하는 상호작용은?",
             "choices":["로봇과 의료진이 함께 설명한다","로봇은 보조 역할만 한다"],"axes":("E","I")},
            {"id":"H3","text":"로봇이 치료 과정을 안내할 때 선호하는 방식은?",
             "choices":["구체적인 치료 단계와 예상 시간을 알려준다","전체적인 치료 목표와 방향성을 설명한다"],"axes":("S","N")},
            {"id":"H4","text":"로봇이 환자의 상태를 모니터링할 때 중시하는 것은?",
             "choices":["정확한 수치와 객관적 데이터","환자의 편안함과 주관적 느낌"],"axes":("T","F")},
            {"id":"H5","text":"로봇이 응급 상황을 감지했을 때 당신의 반응은?",
             "choices":["즉시 의료진에게 연락하고 대응한다","상황을 파악한 후 신중하게 대응한다"],"axes":("J","P")}
        ],
        "도서관": [
            {"id":"L1","text":"도서관에서 로봇이 도서 검색을 도와줄 때 선호하는 방식은?",
             "choices":["구체적인 키워드와 조건을 입력한다","일반적인 주제나 관심사를 말한다"],"axes":("S","N")},
            {"id":"L2","text":"로봇이 독서 추천을 할 때 중시하는 것은?",
             "choices":["인기도와 평점 같은 객관적 지표","개인의 취향과 감정적 연결"],"axes":("T","F")},
            {"id":"L3","text":"도서관에서 로봇과 함께 공부할 때 선호하는 환경은?",
             "choices":["조용하고 집중할 수 있는 개인 공간","다른 사람들과 함께하는 학습 공간"],"axes":("E","I")},
            {"id":"L4","text":"로봇이 도서 대출/반납을 도와줄 때 당신의 스타일은?",
             "choices":["미리 계획하고 한 번에 처리한다","필요할 때마다 개별적으로 처리한다"],"axes":("J","P")},
            {"id":"L5","text":"로봇이 도서관 이용 규칙을 안내할 때 선호하는 방식은?",
             "choices":["명확하고 구체적인 규칙을 제시한다","전체적인 이용 문화와 분위기를 설명한다"],"axes":("S","N")}
        ],
        "쇼핑몰": [
            {"id":"M1","text":"쇼핑몰에서 로봇이 상품을 추천할 때 선호하는 방식은?",
             "choices":["구체적인 상품 정보와 가격을 제공한다","전체적인 스타일과 트렌드를 제안한다"],"axes":("S","N")},
            {"id":"M2","text":"로봇이 할인 정보를 알려줄 때 중시하는 것은?",
             "choices":["정확한 할인율과 절약 금액","특별한 기회와 즐거운 경험"],"axes":("T","F")},
            {"id":"M3","text":"쇼핑몰에서 로봇과 함께 쇼핑할 때 선호하는 방식은?",
             "choices":["미리 목록을 만들고 계획적으로 쇼핑한다","즉흥적으로 발견한 상품을 구매한다"],"axes":("J","P")},
            {"id":"M4","text":"로봇이 매장 위치를 안내할 때 선호하는 설명은?",
             "choices":["구체적인 층수와 위치 번호를 알려준다","전체적인 매장 구조와 분위기를 설명한다"],"axes":("S","N")},
            {"id":"M5","text":"로봇이 고객 서비스를 제공할 때 당신의 반응은?",
             "choices":["즉시 필요한 서비스를 요청한다","먼저 로봇의 서비스 범위를 확인한다"],"axes":("E","I")}
        ],
        "학교": [
            {"id":"S1","text":"학교에서 로봇이 수업을 보조할 때 선호하는 방식은?",
             "choices":["구체적인 학습 목표와 단계를 제시한다","전체적인 학습 흐름과 맥락을 설명한다"],"axes":("S","N")},
            {"id":"S2","text":"로봇이 학생들의 질문에 답할 때 중시하는 것은?",
             "choices":["정확하고 객관적인 정보 제공","학생의 이해도와 감정적 상태 고려"],"axes":("T","F")},
            {"id":"S3","text":"로봇과 함께 그룹 활동을 할 때 선호하는 역할은?",
             "choices":["활발하게 의견을 제시하고 참여한다","조용히 관찰하고 필요할 때만 참여한다"],"axes":("E","I")},
            {"id":"S4","text":"로봇이 과제를 관리할 때 선호하는 방식은?",
             "choices":["명확한 마감일과 체크리스트를 제공한다","유연한 일정과 창의적 접근을 권장한다"],"axes":("J","P")},
            {"id":"S5","text":"로봇이 학교 생활을 안내할 때 당신의 반응은?",
             "choices":["즉시 필요한 정보를 요청한다","전체적인 학교 문화를 먼저 이해한다"],"axes":("E","I")}
        ],
        "공항": [
            {"id":"A1","text":"공항에서 로봇이 수하물을 도와줄 때 선호하는 방식은?",
             "choices":["구체적인 무게와 크기 제한을 확인한다","전체적인 수하물 정책을 이해한다"],"axes":("S","N")},
            {"id":"A2","text":"로봇이 보안 검사를 안내할 때 중시하는 것은?",
             "choices":["정확한 절차와 규정 준수","편안하고 스트레스 없는 경험"],"axes":("T","F")},
            {"id":"A3","text":"로봇이 항공편 정보를 제공할 때 선호하는 설명은?",
             "choices":["구체적인 시간과 게이트 정보를 제공한다","전체적인 여행 일정과 대안을 제시한다"],"axes":("S","N")},
            {"id":"A4","text":"로봇과 함께 공항을 이용할 때 당신의 스타일은?",
             "choices":["미리 계획하고 시간에 맞춰 이동한다","상황에 따라 유연하게 대응한다"],"axes":("J","P")},
            {"id":"A5","text":"로봇이 긴급 상황을 안내할 때 당신의 반응은?",
             "choices":["즉시 지시사항을 따르고 대응한다","상황을 파악한 후 신중하게 판단한다"],"axes":("J","P")}
        ]
    }
    
    # 선택된 장소의 특화 질문 추가
    if location in location_specific_questions:
        return base_questions + location_specific_questions[location]
    else:
        return base_questions

def load_tie_questions(location="일반"):
    """장소별 특화된 타이브레이커 질문 로드"""
    base_tie_questions = {
        "EI": {"axes":("E","I"), "text":"로봇과 함께하는 활동에서 선호하는 환경은?", "choices":["사람들과 함께하는 분위기","조용하고 집중할 수 있는 공간"]},
        "SN": {"axes":("S","N"), "text":"로봇의 미래 기능에 대한 관심은?", "choices":["현재 실용적인 기능에 집중","미래의 혁신적 가능성에 관심"]},
        "TF": {"axes":("T","F"), "text":"로봇과의 관계에서 중시하는 것은?", "choices":["효율성과 성과","감정적 연결과 이해"]},
        "JP": {"axes":("J","P"), "text":"로봇과의 목표 달성에서 선호하는 방식은?", "choices":["계획적이고 체계적인 접근","유연하고 적응적인 방법"]}
    }
    
    # 장소별 특화 타이브레이커 질문
    location_tie_questions = {
        "병원": {
            "EI": {"axes":("E","I"), "text":"병원에서 로봇과 상호작용할 때 선호하는 방식은?", "choices":["다른 환자들과 함께 정보를 공유한다","개인적으로 조용히 상담한다"]},
            "SN": {"axes":("S","N"), "text":"의료 로봇의 정보 제공에서 중시하는 것은?", "choices":["구체적인 검사 결과와 수치","전체적인 건강 상태와 예후"]},
            "TF": {"axes":("T","F"), "text":"로봇이 의료 서비스를 제공할 때 중시하는 것은?", "choices":["정확한 진단과 치료 효과","환자의 편안함과 심리적 안정"]},
            "JP": {"axes":("J","P"), "text":"로봇과의 치료 계획에서 선호하는 방식은?", "choices":["명확한 치료 단계와 일정","상황에 따른 유연한 조정"]}
        },
        "도서관": {
            "EI": {"axes":("E","I"), "text":"도서관에서 로봇과 함께할 때 선호하는 환경은?", "choices":["다른 이용자들과 함께하는 공간","개인적으로 집중할 수 있는 공간"]},
            "SN": {"axes":("S","N"), "text":"로봇의 도서 추천에서 중시하는 것은?", "choices":["구체적인 장르와 저자 정보","전체적인 독서 경험과 감동"]},
            "TF": {"axes":("T","F"), "text":"로봇이 학습을 도와줄 때 중시하는 것은?", "choices":["정확한 정보와 객관적 사실","개인의 관심과 감정적 연결"]},
            "JP": {"axes":("J","P"), "text":"로봇과의 학습 계획에서 선호하는 방식은?", "choices":["체계적인 학습 일정과 목표","자유로운 탐구와 발견"]}
        },
        "쇼핑몰": {
            "EI": {"axes":("E","I"), "text":"쇼핑몰에서 로봇과 상호작용할 때 선호하는 방식은?", "choices":["다른 쇼핑객들과 함께 정보를 공유한다","개인적으로 조용히 상담한다"]},
            "SN": {"axes":("S","N"), "text":"로봇의 상품 추천에서 중시하는 것은?", "choices":["구체적인 상품 정보와 가격","전체적인 스타일과 트렌드"]},
            "TF": {"axes":("T","F"), "text":"로봇이 쇼핑을 도와줄 때 중시하는 것은?", "choices":["효율적인 구매와 절약","즐거운 쇼핑 경험과 만족"]},
            "JP": {"axes":("J","P"), "text":"로봇과의 쇼핑 계획에서 선호하는 방식은?", "choices":["미리 계획하고 목적적으로 쇼핑한다","즉흥적으로 발견한 상품을 구매한다"]}
        },
        "학교": {
            "EI": {"axes":("E","I"), "text":"학교에서 로봇과 함께할 때 선호하는 학습 환경은?", "choices":["다른 학생들과 함께하는 그룹 활동","개인적으로 집중할 수 있는 환경"]},
            "SN": {"axes":("S","N"), "text":"로봇의 학습 안내에서 중시하는 것은?", "choices":["구체적인 학습 목표와 단계","전체적인 학습 흐름과 맥락"]},
            "TF": {"axes":("T","F"), "text":"로봇이 교육을 도와줄 때 중시하는 것은?", "choices":["정확한 지식과 객관적 평가","학생의 관심과 감정적 성장"]},
            "JP": {"axes":("J","P"), "text":"로봇과의 학습 계획에서 선호하는 방식은?", "choices":["체계적인 학습 일정과 평가","자유로운 탐구와 창의적 활동"]}
        },
        "공항": {
            "EI": {"axes":("E","I"), "text":"공항에서 로봇과 상호작용할 때 선호하는 방식은?", "choices":["다른 여행객들과 함께 정보를 공유한다","개인적으로 조용히 상담한다"]},
            "SN": {"axes":("S","N"), "text":"로봇의 여행 안내에서 중시하는 것은?", "choices":["구체적인 시간과 절차 정보","전체적인 여행 경험과 편의"]},
            "TF": {"axes":("T","F"), "text":"로봇이 여행 서비스를 제공할 때 중시하는 것은?", "choices":["정확한 정보와 효율적인 서비스","편안하고 스트레스 없는 경험"]},
            "JP": {"axes":("J","P"), "text":"로봇과의 여행 계획에서 선호하는 방식은?", "choices":["미리 계획하고 시간에 맞춰 진행한다","상황에 따라 유연하게 대응한다"]}
        }
    }
    
    # 선택된 장소의 특화 타이브레이커 질문 사용
    if location in location_tie_questions:
        return location_tie_questions[location]
    else:
        return base_tie_questions

def compute_scores(responses):
    """점수 계산"""
    questions = load_questions(st.session_state.selected_location)
    scores = {axis: 0 for axis in ['E','I','S','N','T','F','J','P']}
    for q in questions:
        choice = responses.get(q['id'])
        if choice is None:
            return None
        pos, neg = q['axes']
        scores[pos if choice == q['choices'][0] else neg] += 1
    return scores

def resolve_ties(scores):
    """동점 해결"""
    tie_questions = load_tie_questions(st.session_state.selected_location)
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
    """MBTI 유형 예측"""
    return ''.join([
        'E' if scores['E'] >= scores['I'] else 'I',
        'S' if scores['S'] >= scores['N'] else 'N',
        'T' if scores['T'] >= scores['F'] else 'F',
        'J' if scores['J'] >= scores['P'] else 'P'
    ])

# 시각화 함수들
def create_score_chart(scores):
    """점수 분포 차트 생성"""
    score_data = {
        '축': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'],
        '점수': [scores['E'], scores['I'], scores['S'], scores['N'], 
                scores['T'], scores['F'], scores['J'], scores['P']]
    }
    score_df = pd.DataFrame(score_data)
    
    fig = px.bar(score_df, x='축', y='점수', 
                title="MBTI 축별 점수",
                color='축',
                color_discrete_map=MBTI_COLORS)
    fig.update_layout(height=300)
    return fig

def create_trend_chart(df, chart_type="line"):
    """트렌드 차트 생성"""
    if df.empty:
        # 빈 데이터일 때 안내 메시지가 포함된 차트 생성
        fig = go.Figure()
        fig.add_annotation(
            text="데이터가 없습니다.<br>먼저 진단을 완료해주세요.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="📊 기간별 MBTI 트렌드",
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    # 날짜 형식 개선
    df['date'] = pd.to_datetime(df['date'])
    daily_mbti = df.groupby(['date', 'mbti']).size().reset_index(name='count')
    
    # 날짜를 더 읽기 쉽게 포맷팅
    daily_mbti['date_formatted'] = daily_mbti['date'].dt.strftime('%Y년 %m월 %d일')
    
    if chart_type == "라인":
        fig = px.line(daily_mbti, x='date', y='count', color='mbti',
                    title="📊 기간별 MBTI 트렌드", color_discrete_map=MBTI_COLORS,
                    hover_data=['date_formatted'])
    elif chart_type == "바":
        fig = px.bar(daily_mbti, x='date', y='count', color='mbti',
                   title="📊 기간별 MBTI 분포", color_discrete_map=MBTI_COLORS,
                   hover_data=['date_formatted'])
    else:
        fig = px.area(daily_mbti, x='date', y='count', color='mbti',
                    title="📊 기간별 MBTI 누적 분포", color_discrete_map=MBTI_COLORS,
                    hover_data=['date_formatted'])
    
    # 레이아웃 개선
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_title="날짜",
        yaxis_title="진단 수",
        xaxis=dict(
            tickformat='%m월 %d일',
            tickmode='auto',
            nticks=min(10, len(daily_mbti['date'].unique()))
        ),
        yaxis=dict(
            tickmode='linear',
            dtick=1
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    # 호버 템플릿 개선
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                     "MBTI: %{fullData.name}<br>" +
                     "진단 수: %{y}<extra></extra>"
    )
    
    return fig

def create_correlation_heatmap(df):
    """상관관계 히트맵 생성"""
    mbti_dummies = pd.get_dummies(df['mbti'])
    corr = mbti_dummies.corr()
    fig = px.imshow(corr, title="MBTI 유형 간 상관행렬", 
                   aspect="auto", color_continuous_scale="RdBu")
    return fig, corr

# MBTI 가이드 데이터
def load_guide_data(location="일반"):
    """장소별 특화된 MBTI 가이드 데이터 로드"""
    base_guide = {
        "ENFJ": {
            "description": "리더십과 공감 능력을 겸비한 타입입니다. 다른 사람의 성장을 돕고, 팀의 조화를 중시합니다.",
            "hri_style": "공감적 리더십, 격려와 성장 지향, 팀워크 중시",
            "examples": [
                "안녕하세요! 오늘 어떤 도움이 필요하신지 편하게 말씀해 주세요.",
                "함께 이 문제를 해결해보는 건 어떨까요? 당신의 생각이 궁금해요."
            ]
        },
        "ENTJ": {
            "description": "전략적 사고와 효율성을 중시하는 타입입니다. 명확한 목표 설정과 체계적인 접근을 선호합니다.",
            "hri_style": "전략적 사고, 효율성 중시, 목표 지향적",
            "examples": [
                "목표 달성을 위해 체계적으로 안내해 드리겠습니다. 단계별로 진행하시죠.",
                "현재 상황을 분석한 결과, 이 방법이 가장 효율적일 것 같습니다."
            ]
        },
        "ENTP": {
            "description": "창의적이고 혁신적인 사고를 가진 타입입니다. 새로운 아이디어와 도전을 즐깁니다.",
            "hri_style": "혁신적 접근, 창의적 해결책, 도전 지향적",
            "examples": [
                "새로운 관점에서 이 문제를 바라보는 건 어떨까요?",
                "기존 방식을 개선할 수 있는 창의적인 방법을 제안해 드릴게요."
            ]
        },
        "ENFP": {
            "description": "열정적이고 창의적인 타입입니다. 가능성과 새로운 경험을 추구합니다.",
            "hri_style": "열정적 소통, 창의적 영감, 가능성 추구",
            "examples": [
                "정말 흥미로운 아이디어네요! 더 자세히 들어보고 싶어요.",
                "새로운 가능성을 함께 탐색해보는 건 어떨까요?"
            ]
        },
        "ESFJ": {
            "description": "협력적이고 실용적인 타입입니다. 다른 사람의 필요를 돌보고 조화를 추구합니다.",
            "hri_style": "협력적 지원, 실용적 도움, 조화 중시",
            "examples": [
                "어떤 도움이 필요하신지 말씀해 주세요. 함께 해결해보겠습니다.",
                "모두가 편안하게 이용할 수 있도록 도와드릴게요."
            ]
        },
        "ESFP": {
            "description": "즉흥적이고 친근한 타입입니다. 현재의 즐거움과 실용적 해결책을 중시합니다.",
            "hri_style": "즉흥적 상호작용, 친근한 소통, 실용적 해결",
            "examples": [
                "지금 당장 도움이 필요하시군요! 바로 해결해드릴게요.",
                "편하게 말씀해 주세요. 함께 즐겁게 해결해보죠."
            ]
        },
        "ESTJ": {
            "description": "체계적이고 책임감 있는 타입입니다. 규칙과 효율성을 중시합니다.",
            "hri_style": "체계적 관리, 책임감 있는 안내, 효율성 중시",
            "examples": [
                "규정에 따라 체계적으로 안내해 드리겠습니다.",
                "효율적으로 진행할 수 있도록 단계별로 도와드릴게요."
            ]
        },
        "ESTP": {
            "description": "실용적이고 적응력이 뛰어난 타입입니다. 현재 상황에 맞는 해결책을 찾습니다.",
            "hri_style": "실용적 해결, 적응적 대응, 즉시 실행",
            "examples": [
                "현재 상황에 맞는 실용적인 해결책을 제안해 드릴게요.",
                "바로 실행할 수 있는 방법을 알려드리겠습니다."
            ]
        },
        "INFJ": {
            "description": "직관적이고 이상주의적인 타입입니다. 깊은 통찰력과 창의성을 가집니다.",
            "hri_style": "직관적 이해, 깊은 통찰, 창의적 접근",
            "examples": [
                "더 깊이 있는 이해를 위해 함께 탐색해보는 건 어떨까요?",
                "본질적인 문제를 찾아 해결해보겠습니다."
            ]
        },
        "INFP": {
            "description": "이상주의적이고 창의적인 타입입니다. 개인의 가치와 의미를 중시합니다.",
            "hri_style": "이상주의적 접근, 창의적 영감, 개인적 가치 중시",
            "examples": [
                "당신만의 특별한 관점이 궁금해요. 함께 이야기해보죠.",
                "의미 있는 경험을 만들어보는 건 어떨까요?"
            ]
        },
        "INTJ": {
            "description": "전략적이고 독창적인 사고를 가진 타입입니다. 장기적 비전과 효율성을 추구합니다.",
            "hri_style": "전략적 계획, 독창적 해결책, 장기적 비전",
            "examples": [
                "장기적인 관점에서 최적의 해결책을 제안해 드릴게요.",
                "전략적으로 접근하여 효율적으로 해결해보겠습니다."
            ]
        },
        "INTP": {
            "description": "논리적이고 분석적인 타입입니다. 복잡한 문제를 해결하는 것을 즐깁니다.",
            "hri_style": "논리적 분석, 복잡한 문제 해결, 정확성 중시",
            "examples": [
                "논리적으로 분석해보니 이런 해결책이 가장 적합할 것 같아요.",
                "복잡한 문제를 단계별로 분석해서 해결해보겠습니다."
            ]
        },
        "ISFJ": {
            "description": "신중하고 헌신적인 타입입니다. 실용적이고 안정적인 해결책을 제공합니다.",
            "hri_style": "신중한 지원, 실용적 해결, 안정적 서비스",
            "examples": [
                "신중하게 검토한 후 안전하고 실용적인 방법을 제안해 드릴게요.",
                "안정적으로 도움을 드릴 수 있도록 체계적으로 진행하겠습니다."
            ]
        },
        "ISFP": {
            "description": "예술적이고 실용적인 타입입니다. 현재의 경험과 개인의 가치를 중시합니다.",
            "hri_style": "예술적 접근, 실용적 해결, 개인적 경험 중시",
            "examples": [
                "당신만의 특별한 방식으로 해결해보는 건 어떨까요?",
                "현재의 경험을 최대한 활용해서 도와드릴게요."
            ]
        },
        "ISTJ": {
            "description": "신뢰할 수 있고 체계적인 타입입니다. 규칙과 정확성을 중시합니다.",
            "hri_style": "신뢰할 수 있는 안내, 체계적 관리, 정확성 중시",
            "examples": [
                "규정에 따라 정확하고 신뢰할 수 있는 정보를 제공해 드릴게요.",
                "체계적으로 진행하여 안전하게 도와드리겠습니다."
            ]
        },
        "ISTP": {
            "description": "실용적이고 분석적인 타입입니다. 문제 해결과 효율성을 중시합니다.",
            "hri_style": "실용적 해결, 분석적 접근, 효율성 중시",
            "examples": [
                "문제를 분석해서 실용적인 해결책을 제안해 드릴게요.",
                "효율적으로 진행할 수 있도록 분석적으로 도와드리겠습니다."
            ]
        }
    }
    
    # 장소별 특화 가이드
    location_guides = {
        "병원": {
            "ENFJ": {"hri_style": "공감적 의료 서비스, 환자 중심적 접근, 치료팀 협력"},
            "ENTJ": {"hri_style": "전략적 치료 계획, 효율적 의료 관리, 체계적 진료"},
            "INFJ": {"hri_style": "직관적 진단, 깊은 환자 이해, 개인화된 치료"},
            "INTJ": {"hri_style": "전략적 의료 계획, 혁신적 치료 방법, 장기적 건강 관리"}
        },
        "도서관": {
            "ENFJ": {"hri_style": "공감적 학습 지원, 독서 문화 조성, 이용자 성장 도움"},
            "ENTJ": {"hri_style": "전략적 학습 계획, 효율적 정보 관리, 체계적 교육"},
            "INFJ": {"hri_style": "직관적 독서 추천, 깊은 지식 탐구, 개인화된 학습"},
            "INTJ": {"hri_style": "전략적 학습 설계, 혁신적 교육 방법, 장기적 지식 구축"}
        },
        "쇼핑몰": {
            "ENFJ": {"hri_style": "공감적 고객 서비스, 쇼핑 경험 향상, 고객 만족 중시"},
            "ENTJ": {"hri_style": "전략적 쇼핑 안내, 효율적 구매 지원, 체계적 서비스"},
            "INFJ": {"hri_style": "직관적 상품 추천, 깊은 고객 이해, 개인화된 서비스"},
            "INTJ": {"hri_style": "전략적 쇼핑 계획, 혁신적 서비스 방법, 장기적 고객 관계"}
        },
        "학교": {
            "ENFJ": {"hri_style": "공감적 교육 지원, 학생 성장 도움, 학습 환경 조성"},
            "ENTJ": {"hri_style": "전략적 학습 계획, 효율적 교육 관리, 체계적 학습"},
            "INFJ": {"hri_style": "직관적 학습 안내, 깊은 학생 이해, 개인화된 교육"},
            "INTJ": {"hri_style": "전략적 교육 설계, 혁신적 학습 방법, 장기적 지식 구축"}
        },
        "공항": {
            "ENFJ": {"hri_style": "공감적 여행 서비스, 편안한 여행 경험, 여행자 안내"},
            "ENTJ": {"hri_style": "전략적 여행 계획, 효율적 여행 관리, 체계적 서비스"},
            "INFJ": {"hri_style": "직관적 여행 안내, 깊은 여행자 이해, 개인화된 서비스"},
            "INTJ": {"hri_style": "전략적 여행 설계, 혁신적 서비스 방법, 장기적 여행 계획"}
        }
    }
    
    # 선택된 장소의 특화 가이드 적용
    if location in location_guides:
        for mbti_type in base_guide:
            if mbti_type in location_guides[location]:
                base_guide[mbti_type]["hri_style"] = location_guides[location][mbti_type]["hri_style"]
    
    return base_guide

def check_existing_diagnosis(user_id, robot_id):
    """같은 사용자-로봇 조합의 최근 진단 확인"""
    try:
        # 최근 24시간 내 같은 사용자-로봇 조합의 진단 확인
        yesterday = datetime.now(pytz.timezone("Asia/Seoul")) - timedelta(hours=24)
        res = supabase.table("responses").select("*").eq("user_id", user_id).eq("robot_id", robot_id).gte("timestamp", yesterday.isoformat()).execute()
        
        if res.data:
            return True, res.data[0]  # 최근 진단이 있음
        return False, None
    except Exception as e:
        st.error(f"진단 확인 중 오류: {e}")
        return False, None

def generate_diagnosis_id():
    """고유한 진단 세션 ID 생성"""
    return f"diagnosis_{int(time.time())}_{st.session_state.user_id}_{st.session_state.robot_id}"

def save_response_with_session(diagnosis_data):
    """진단 세션 ID를 포함한 응답 데이터 저장"""
    try:
        # 입력값 정제
        diagnosis_data["user_id"] = sanitize_input(diagnosis_data["user_id"])
        diagnosis_data["robot_id"] = sanitize_input(diagnosis_data["robot_id"])
        
        # 유효성 검증
        user_valid, user_msg = validate_user_id(diagnosis_data["user_id"])
        robot_valid, robot_msg = validate_robot_id(diagnosis_data["robot_id"])
        
        if not user_valid:
            st.error(f"사용자 ID 오류: {user_msg}")
            return False
        
        if not robot_valid:
            st.error(f"로봇 ID 오류: {robot_msg}")
            return False
        
        # diagnosis_session_id가 있으면 중복 확인, 없으면 기본 저장
        if "diagnosis_session_id" in diagnosis_data:
            try:
                # 중복 진단 세션 확인
                existing = supabase.table("responses").select("diagnosis_session_id").eq("diagnosis_session_id", diagnosis_data["diagnosis_session_id"]).execute()
                if existing.data:
                    st.warning("이미 저장된 진단 세션입니다.")
                    return False
            except Exception as e:
                # diagnosis_session_id 컬럼이 없는 경우 무시하고 계속 진행
                st.info("진단 세션 ID 기능이 비활성화되어 있습니다.")
        
        # diagnosis_session_id 제거하고 기본 필드만 저장
        save_data = {
            "user_id": diagnosis_data["user_id"],
            "gender": diagnosis_data["gender"],
            "age_group": diagnosis_data["age_group"],
            "job": diagnosis_data["job"],
            "robot_id": diagnosis_data["robot_id"],
            "responses": diagnosis_data["responses"],
            "mbti": diagnosis_data["mbti"],
            "scores": diagnosis_data["scores"],
            "timestamp": diagnosis_data["timestamp"]
        }
        
        supabase.table("responses").insert(save_data).execute()
        return True
    except Exception as e:
        st.error(f"응답 저장 실패: {e}")
        return False

# 초기화
init_session_state()
setup_styles()
guide_data = load_guide_data(st.session_state.get('selected_location', '일반'))

# 사용자 ID 확인
if not st.session_state.user_id:
    st.markdown("# 🤖 로봇 MBTI 진단 시스템")
    st.markdown("### 사용자 ID를 입력해주세요")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_id = st.text_input("사용자 ID", placeholder="예: 김철수, 연구실A, 팀1", key="user_id_input")
        if st.button("시작하기"):
            user_valid, user_msg = validate_user_id(user_id)
            if user_valid:
                st.session_state.user_id = user_id
                st.success(f"사용자 ID '{user_id}'로 설정되었습니다.")
                st.rerun()
            else:
                st.error(f"사용자 ID 오류: {user_msg}")
        
        with st.expander("사용자 ID 가이드"):
            st.info("""
            **사용자 ID 예시:**
            - 개인: 김철수, 홍길동, 연구자A
            - 팀: 연구실A, 팀1, 그룹B
            - 프로젝트: 프로젝트A, 실험1, 테스트팀
            
            **규칙:**
            - 2-20자 이내
            - 영문, 숫자, 한글, 공백, 언더스코어(_), 하이픈(-) 사용 가능
            - 특수문자 제한 (보안상)
            """)
    st.stop()

# 로그인된 경우 USER_ID 설정
USER_ID = st.session_state.user_id

# 사이드바 표시
def show_sidebar():
    """사이드바 표시"""
    with st.sidebar:
        st.header("👤 사용자/로봇 정보")
        show_user_input()
        show_user_profile()
        show_robot_management()
        show_admin_section()

def show_user_input():
    """사용자 ID 입력"""
    st.subheader("🆔 사용자 ID")
    
    # 현재 사용자 ID 표시
    if st.session_state.user_id:
        st.success(f"현재 사용자: **{st.session_state.user_id}**")
        if st.button("사용자 변경"):
            st.session_state.user_id = None
            st.rerun()
    else:
        user_id = st.text_input("사용자 ID 입력", placeholder="예: 김철수, 연구실A, 팀1")
        if st.button("확인"):
            user_valid, user_msg = validate_user_id(user_id)
            if user_valid:
                st.session_state.user_id = user_id
                st.success(f"사용자 ID '{user_id}'로 설정되었습니다.")
                st.rerun()
            else:
                st.error(f"사용자 ID 오류: {user_msg}")

def show_user_profile():
    """사용자 프로필 표시"""
    if not st.session_state.user_id:
        st.warning("먼저 사용자 ID를 입력해주세요.")
        return
    
    st.divider()
    st.subheader("👤 사용자 프로필")
    
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

def show_robot_management():
    """로봇 관리 표시"""
    if not st.session_state.user_id:
        st.warning("먼저 사용자 ID를 입력해주세요.")
        return
    
    st.divider()
    st.subheader("🤖 로봇 ID 관리")
    
    # 데이터베이스 상태 확인
    db_status = "❌ 알 수 없는 오류"
    try:
        # 테이블 존재 여부 확인
        test_result = supabase.table("user_robots").select("id").limit(1).execute()
        db_status = "✅ 데이터베이스 연결됨"
    except Exception as e:
        error_msg = str(e).lower()
        if "does not exist" in error_msg or "relation" in error_msg:
            db_status = "⚠️ user_robots 테이블이 존재하지 않습니다"
            st.warning("""
            **user_robots 테이블이 생성되지 않았습니다.**
            
            해결 방법:
            1. Supabase 대시보드 → SQL Editor
            2. `create_user_robots_table.sql` 파일의 내용을 실행
            3. 또는 관리자에게 문의
            """)
        elif "permission" in error_msg:
            db_status = "❌ 권한 오류"
        elif "connection" in error_msg:
            db_status = "❌ 연결 오류"
        else:
            db_status = f"❌ 데이터베이스 오류: {str(e)[:50]}"
    
    st.caption(f"상태: {db_status}")
    
    # 로봇 목록 불러오기 (로컬 + 데이터베이스)
    robot_opts = list(st.session_state.robot_list)
    
    # 데이터베이스에서 로봇 목록 불러오기 시도
    try:
        db_robots = load_user_robots(st.session_state.user_id)
        for robot in db_robots:
            if robot not in robot_opts:
                robot_opts.append(robot)
    except Exception as e:
        st.info(f"데이터베이스에서 로봇 목록을 불러올 수 없습니다: {str(e)[:50]}")
    
    # 새 로봇 등록
    col1, col2 = st.columns([3, 1])
    with col1:
        new_robot = st.text_input("새 로봇 ID 등록", placeholder="예: 내식기1, 로봇A")
    with col2:
        if st.button("➕ 등록"):
            if not new_robot.strip():
                st.warning("로봇 ID를 입력해주세요.")
            else:
                robot_valid, robot_msg = validate_robot_id(new_robot)
                if robot_valid:
                    if new_robot.strip() not in robot_opts:
                        if save_robot(st.session_state.user_id, new_robot.strip()):
                            robot_opts.insert(0, new_robot.strip())
                            st.session_state.robot_list = robot_opts
                            st.success(f"✅ '{new_robot.strip()}' 등록 완료!")
                            st.rerun()
                        else:
                            st.error("로봇 등록에 실패했습니다. 데이터베이스 상태를 확인해주세요.")
                    else:
                        st.warning("이미 등록된 로봇입니다.")
                else:
                    st.error(f"로봇 ID 오류: {robot_msg}")
    
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
                try:
                    # 데이터베이스에서 삭제 시도
                    supabase.table("user_robots").delete().eq("user_id", st.session_state.user_id).eq("robot_name", delete_robot).execute()
                    robot_opts.remove(delete_robot)
                    st.session_state.robot_list = robot_opts
                    if st.session_state.robot_id == delete_robot:
                        st.session_state.robot_id = robot_opts[0]
                    st.success(f"✅ '{delete_robot}' 삭제 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"로봇 삭제에 실패했습니다: {str(e)[:50]}")
        else:
            st.info("로봇은 최소 1개 이상 유지해야 합니다.")
    else:
        st.warning("등록된 로봇이 없습니다. 새 로봇을 등록해주세요.")

def show_admin_section():
    """관리자 섹션 표시"""
    st.divider()
    st.subheader("🔧 관리자 로그인")
    
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

# 메인 콘텐츠 표시
def show_main_content():
    """메인 콘텐츠 표시"""
    if not st.session_state.user_id:
        st.warning("사이드바에서 사용자 ID를 입력해주세요.")
        return
    
    page = st.session_state.page
    
    if page == 1:
        show_diagnosis_page()
    elif page == 2:
        show_results_page()
    elif page == 3:
        show_analytics_page()

def show_diagnosis_page():
    """진단 페이지 표시"""
    st.header("1️⃣ MBTI 기반 HRI UX 진단")
    
    # 사용자 정보 표시
    st.info(f"👤 **사용자**: {st.session_state.user_id} | 🤖 **로봇**: {st.session_state.robot_id}")
    
    # 중복 진단 확인
    has_recent_diagnosis, recent_diagnosis = check_existing_diagnosis(st.session_state.user_id, st.session_state.robot_id)
    
    if has_recent_diagnosis:
        st.warning(f"⚠️ 최근 24시간 내에 이미 '{st.session_state.robot_id}'에 대한 진단이 완료되었습니다.")
        st.info(f"마지막 진단 시간: {recent_diagnosis['timestamp']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("새로운 진단 진행", type="primary"):
                # 새로운 진단 세션 시작
                st.session_state.current_diagnosis_id = generate_diagnosis_id()
                st.session_state.saved_result = False
                st.rerun()
        with col2:
            if st.button("결과 페이지로 이동"):
                st.session_state.page = 2
                st.rerun()
        return
    
    # 진단 장소 선택
    st.subheader("🏢 진단 장소 선택")
    location_options = ["일반", "병원", "도서관", "쇼핑몰", "학교", "공항"]
    selected_location = st.selectbox(
        "진단할 장소를 선택하세요",
        location_options,
        index=0,
        help="장소별로 다른 설문 내용이 제공됩니다"
    )
    
    # 선택된 장소를 세션에 저장
    st.session_state.selected_location = selected_location
    
    st.info(f"📋 {selected_location} 환경 진단 (총 {len(load_questions(selected_location))}개 질문)")
    
    st.divider()
    
    consent = st.checkbox("익명 데이터 분석 활용에 동의합니다.", value=True)
    if not consent:
        st.warning("진단 시작엔 동의가 필요합니다!")
        st.stop()
    
    # 설문 표시
    responses = show_survey()
    
    # 결과 보기 버튼
    if st.button("🎯 결과 보기", type="primary", use_container_width=True):
        if len(responses) < len(load_questions(selected_location)):
            st.warning("모든 문항에 답변해주세요!")
        else:
            # 새로운 진단 세션 시작
            if not st.session_state.current_diagnosis_id:
                st.session_state.current_diagnosis_id = generate_diagnosis_id()
            
            st.session_state.page = 2
            st.rerun()

def show_survey():
    """설문 표시"""
    responses = {}
    current_questions = load_questions(st.session_state.selected_location)
    
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
    return responses

def show_results_page():
    """결과 페이지 표시"""
    st.header(f"2️⃣ [{st.session_state.robot_id}] 진단 결과·피드백")
    
    with st.spinner("🔍 MBTI 분석 중..."):
        time.sleep(1)
    
    responses = st.session_state.get('responses', {})
    profile = st.session_state.user_profile
    robot_id = st.session_state.robot_id
    user_id = st.session_state.user_id
    
    # 응답이 없으면 이전 진단 데이터 확인
    if not responses:
        has_recent_diagnosis, recent_diagnosis = check_existing_diagnosis(user_id, robot_id)
        if has_recent_diagnosis:
            st.info("📋 이전 진단 결과를 표시합니다.")
            mbti = recent_diagnosis['mbti']
            scores = recent_diagnosis['scores']
            display_results(mbti, scores, robot_id)
            return
        else:
            st.warning("진단 데이터가 없습니다. 먼저 진단을 완료해주세요.")
            st.session_state.page = 1
            st.rerun()
            return
    
    scores = compute_scores(responses)
    
    if scores is None:
        st.warning("모든 문항에 답해주세요.")
    else:
        scores = resolve_ties(scores)
        mbti = predict_type(scores)
        
        # 중복 저장 방지: 현재 진단 세션에서만 저장
        if not st.session_state.get('saved_result', False) and st.session_state.current_diagnosis_id:
            profile_with_location = profile.copy()
            profile_with_location['location'] = st.session_state.get('selected_location', '일반')
            
            # 진단 세션 ID를 포함하여 저장
            diagnosis_data = {
                "user_id": user_id,
                "gender": profile["gender"],
                "age_group": profile["age_group"],
                "job": profile["job"],
                "robot_id": robot_id,
                "responses": responses,
                "mbti": mbti,
                "scores": scores,
                "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
                "diagnosis_session_id": st.session_state.current_diagnosis_id  # 세션 ID 추가
            }
            
            # 저장 성공 시에만 saved_result를 True로 설정
            if save_response_with_session(diagnosis_data):
                st.session_state['saved_result'] = True
                st.success("✅ 진단 결과가 성공적으로 저장되었습니다!")
            else:
                st.error("❌ 진단 결과 저장에 실패했습니다.")
        
        display_results(mbti, scores, robot_id)

def display_results(mbti, scores, robot_id):
    """결과 표시"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ## 🎯 진단 결과
        
        ### 👤 사용자: **{st.session_state.user_id}**
        ### 🤖 로봇: **{robot_id}**
        ### 🏢 장소: **{st.session_state.get('selected_location', '일반')}**
        ### 🧠 MBTI 유형: **{mbti}**
        """)
        
        # MBTI 유형별 색상 표시 (크기 조정)
        mbti_color = MBTI_COLORS.get(mbti, '#CCCCCC')
        st.markdown(f"""
        <div style="background-color: {mbti_color}; padding: 15px; border-radius: 10px; text-align: center; color: white; margin: 10px 0;">
            <h3 style="margin: 0; font-size: 1.5rem;">🎨 {mbti}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 점수 분포")
        fig = create_score_chart(scores)
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # 장소별 특화 조언
    location = st.session_state.get('selected_location', '일반')
    if location != '일반':
        st.subheader(f"🏢 {location} 환경 특화 조언")
        location_advice = {
            "병원": f"**{mbti}** 유형으로서 병원 환경에서는 {guide.get('hri_style', '효율적이고 신뢰할 수 있는 서비스')}를 제공하는 것이 좋습니다. 환자의 안전과 편안함을 최우선으로 고려하세요.",
            "도서관": f"**{mbti}** 유형으로서 도서관 환경에서는 {guide.get('hri_style', '조용하고 집중할 수 있는 학습 지원')}을 제공하는 것이 좋습니다. 지식 탐구와 학습 환경 조성에 중점을 두세요.",
            "쇼핑몰": f"**{mbti}** 유형으로서 쇼핑몰 환경에서는 {guide.get('hri_style', '친근하고 도움이 되는 고객 서비스')}를 제공하는 것이 좋습니다. 고객의 쇼핑 경험 향상에 집중하세요.",
            "학교": f"**{mbti}** 유형으로서 학교 환경에서는 {guide.get('hri_style', '교육적이고 성장 지향적인 학습 지원')}을 제공하는 것이 좋습니다. 학생의 학습과 성장을 돕는 데 중점을 두세요.",
            "공항": f"**{mbti}** 유형으로서 공항 환경에서는 {guide.get('hri_style', '효율적이고 안전한 여행 서비스')}를 제공하는 것이 좋습니다. 여행자의 편의와 안전을 최우선으로 고려하세요."
        }
        st.info(location_advice.get(location, ""))
    
    # 다운로드 및 다음 단계
    st.subheader("💾 결과 저장")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📥 결과 CSV 다운로드", 
                         pd.DataFrame([{'사용자': st.session_state.user_id, '로봇': robot_id, 'MBTI': mbti, '날짜': datetime.now().strftime('%Y-%m-%d')}]).to_csv(index=False), 
                         f"{st.session_state.user_id}_{robot_id}_{mbti}.csv")
    with col2:
        if st.button("📊 통계/히스토리 대시보드 이동", type="primary", use_container_width=True):
            st.session_state.page = 3
            st.rerun()
    with col3:
        if st.button("🔄 새로운 진단 시작", use_container_width=True):
            reset_diagnosis_session()
            st.session_state.page = 1
            st.rerun()

def show_analytics_page():
    """분석 페이지 표시"""
    st.header("3️⃣ 전체 통계 · 로봇 이력 · 집단분석(통합)")
    
    df = load_responses()
    if df.empty:
        st.info("아직 데이터가 없습니다.")
        if st.button("진단 첫화면으로 돌아가기"):
            st.session_state.page = 1
            st.rerun()
        return
    
    # 중복 데이터 제거 (같은 사용자, 같은 로봇, 같은 시간대의 중복 진단)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    # 중복 제거: 같은 사용자-로봇 조합에서 가장 최근 진단만 유지
    df_cleaned = df.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'robot_id'], 
        keep='last'
    )
    
    # 중복 제거 결과 표시
    if len(df) != len(df_cleaned):
        st.info(f"📊 중복 진단 데이터가 정리되었습니다. (전체: {len(df)} → 정리 후: {len(df_cleaned)})")
    
    df = df_cleaned
    
    # 탭으로 구분
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 전체 트렌드", "📈 집단별 분석", "🤖 로봇 이력", 
        "🧠 고급 분석", "📋 데이터 관리", "🔧 관리자 관리"
    ])
    
    with tab1:
        show_trend_analysis(df)
    
    with tab2:
        show_group_analysis(df)
    
    with tab3:
        show_robot_history(df)
    
    with tab4:
        show_advanced_analysis(df)
    
    with tab5:
        show_data_management(df)
    
    with tab6:
        show_admin_data_management(df)
    
    if st.button("진단 첫화면으로 돌아가기"):
        st.session_state.page = 1
        st.rerun()

def show_trend_analysis(df):
    """트렌드 분석 표시"""
    st.subheader("📊 기간별 MBTI 트렌드")
    min_date, max_date = df['date'].min(), df['date'].max()
    
    # chart_type 변수를 먼저 초기화
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
        fig = create_trend_chart(df_period, chart_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # 요약 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 진단 수", len(df_period))
        with col2:
            st.metric("MBTI 유형 수", df_period['mbti'].nunique())
        with col3:
            st.metric("가장 많은 유형", df_period['mbti'].mode().iloc[0] if not df_period['mbti'].mode().empty else "N/A")

def show_group_analysis(df):
    """집단별 분석 표시"""
    st.subheader("📈 집단별 MBTI 분포 분석")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        group_col = st.selectbox("분포 분석 기준", ["gender", "age_group", "job", "robot_id"])
        chart_style = st.selectbox("차트 스타일", ["바 차트", "파이 차트", "히트맵"])
    
    with col2:
        group_df = df.groupby([group_col, "mbti"]).size().unstack(fill_value=0)
        
        if chart_style == "바 차트":
            fig = px.bar(group_df, title=f"{group_col}별 MBTI 분포", 
                       color_discrete_map=MBTI_COLORS)
            fig.update_layout(height=400, xaxis_title=group_col, yaxis_title="진단 수")
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_style == "파이 차트":
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
                          marker_colors=[MBTI_COLORS.get(mbti, '#CCCCCC') for mbti in labels]),
                    row=row, col=col
                )
            
            fig.update_layout(height=300 * rows, title_text=f"{group_col}별 MBTI 분포")
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # 히트맵
            fig = px.imshow(group_df, title=f"{group_col}별 MBTI 히트맵",
                          aspect="auto", color_continuous_scale="Viridis")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_robot_history(df):
    """로봇 이력 표시"""
    st.subheader(f"🤖 '{st.session_state.robot_id}'의 MBTI 변화 히스토리")
    
    # 현재 사용자의 로봇 데이터만 필터링
    bot_records = df[(df['user_id']==st.session_state.user_id) & (df['robot_id']==st.session_state.robot_id)].sort_values("timestamp")
    
    if not bot_records.empty:
        # 날짜 형식 개선
        bot_records['timestamp'] = pd.to_datetime(bot_records['timestamp'])
        bot_records['date_formatted'] = bot_records['timestamp'].dt.strftime('%Y년 %m월 %d일')
        bot_records['time_formatted'] = bot_records['timestamp'].dt.strftime('%H:%M')
        
        # 타임라인 차트 개선
        fig = px.scatter(
            bot_records, 
            x='timestamp', 
            y='mbti', 
            title=f"📈 '{st.session_state.robot_id}' MBTI 변화 타임라인",
            color='mbti', 
            color_discrete_map=MBTI_COLORS,
            size=[25] * len(bot_records),
            hover_data=['date_formatted', 'time_formatted', 'gender', 'age_group', 'job']
        )
        
        # 레이아웃 개선
        fig.update_layout(
            height=500,
            xaxis_title="날짜",
            yaxis_title="MBTI 유형",
            xaxis=dict(
                tickformat='%m월 %d일',
                tickmode='auto',
                nticks=min(10, len(bot_records)),
                tickangle=45
            ),
            yaxis=dict(
                categoryorder='array',
                categoryarray=['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                             'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
            ),
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=80)
        )
        
        # 마커 스타일 개선
        fig.update_traces(
            marker=dict(
                size=20,
                line=dict(width=2, color='white')
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "시간: %{customdata[1]}<br>" +
                         "MBTI: %{y}<br>" +
                         "성별: %{customdata[2]}<br>" +
                         "연령대: %{customdata[3]}<br>" +
                         "직업: %{customdata[4]}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 이력 테이블
        st.subheader("📋 상세 진단 이력")
        
        # 테이블 데이터 준비
        history_df = bot_records[["date_formatted", "time_formatted", "mbti", "gender", "age_group", "job"]].copy()
        history_df.columns = ["날짜", "시간", "MBTI", "성별", "연령대", "직업"]
        
        # MBTI 색상 적용
        def color_mbti(val):
            color = MBTI_COLORS.get(val, '#CCCCCC')
            return f'background-color: {color}; color: white; font-weight: bold; text-align: center;'
        
        styled_df = history_df.style.map(color_mbti, subset=['MBTI'])
        st.dataframe(styled_df, use_container_width=True)
        
        # 통계 요약
        st.subheader("📊 진단 통계")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 진단 수", len(bot_records))
        with col2:
            st.metric("MBTI 유형 수", bot_records['mbti'].nunique())
        with col3:
            most_common = bot_records['mbti'].mode().iloc[0] if not bot_records['mbti'].mode().empty else "N/A"
            st.metric("가장 많은 유형", most_common)
        with col4:
            days_span = (bot_records['timestamp'].max() - bot_records['timestamp'].min()).days + 1
            st.metric("진단 기간", f"{days_span}일")
        
        # MBTI 변화 분석
        if len(bot_records) > 1:
            st.subheader("🔄 MBTI 변화 분석")
            
            # 변화 패턴 분석
            changes = []
            for i in range(1, len(bot_records)):
                prev_mbti = bot_records.iloc[i-1]['mbti']
                curr_mbti = bot_records.iloc[i]['mbti']
                if prev_mbti != curr_mbti:
                    changes.append({
                        'from': prev_mbti,
                        'to': curr_mbti,
                        'date': bot_records.iloc[i]['date_formatted']
                    })
            
            if changes:
                st.info(f"총 {len(changes)}번의 MBTI 변화가 있었습니다.")
                
                # 변화 차트
                if len(changes) > 0:
                    change_df = pd.DataFrame(changes)
                    fig_changes = px.scatter(
                        change_df,
                        x='date',
                        y='from',
                        color='to',
                        title="MBTI 변화 패턴",
                        color_discrete_map=MBTI_COLORS
                    )
                    fig_changes.update_layout(height=300)
                    st.plotly_chart(fig_changes, use_container_width=True)
            else:
                st.success("MBTI 유형이 일관되게 유지되고 있습니다.")
    else:
        st.info(f"로봇 '{st.session_state.robot_id}'의 진단 이력이 없습니다.")

def create_mbti_network(df):
    """MBTI 네트워크 분석 생성"""
    if len(df) < 2:
        return None, "네트워크 분석을 위해서는 최소 2개의 진단 데이터가 필요합니다."
    
    # MBTI 유형 간 관계 분석
    mbti_counts = df['mbti'].value_counts()
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 노드 추가 (MBTI 유형)
    for mbti in mbti_counts.index:
        G.add_node(mbti, size=mbti_counts[mbti], color=MBTI_COLORS.get(mbti, '#CCCCCC'))
    
    # 엣지 추가 (공통 특성 기반)
    mbti_axes = {
        'E': ['ENFJ', 'ENTJ', 'ENTP', 'ENFP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'],
        'I': ['INFJ', 'INTJ', 'INTP', 'INFP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
        'S': ['ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
        'N': ['ENFJ', 'ENTJ', 'ENTP', 'ENFP', 'INFJ', 'INTJ', 'INTP', 'INFP'],
        'T': ['ENTJ', 'ENTP', 'ESTJ', 'ESTP', 'INTJ', 'INTP', 'ISTJ', 'ISTP'],
        'F': ['ENFJ', 'ENFP', 'ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'],
        'J': ['ENFJ', 'ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'],
        'P': ['ENTP', 'ENFP', 'ESFP', 'ESTP', 'INTP', 'INFP', 'ISFP', 'ISTP']
    }
    
    # 공통 특성을 가진 MBTI 유형들 간에 엣지 추가
    for axis, mbti_types in mbti_axes.items():
        for i, mbti1 in enumerate(mbti_types):
            for mbti2 in mbti_types[i+1:]:
                if mbti1 in G.nodes and mbti2 in G.nodes:
                    if G.has_edge(mbti1, mbti2):
                        G[mbti1][mbti2]['weight'] += 1
                    else:
                        G.add_edge(mbti1, mbti2, weight=1, axis=axis)
    
    # 네트워크 레이아웃 계산
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Plotly 네트워크 그래프 생성
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(size=[], color=[], line=dict(width=2)))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple([G.nodes[node]['color']])
        node_trace['marker']['size'] += tuple([G.nodes[node]['size'] * 5 + 10])
        node_trace['text'] += tuple([f"{node}<br>진단 수: {G.nodes[node]['size']}"])
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='🧠 MBTI 네트워크 분석',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    # 네트워크 통계 계산
    stats = {
        'total_nodes': len(G.nodes),
        'total_edges': len(G.edges),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'connected_components': nx.number_connected_components(G)
    }
    
    return fig, stats

def show_advanced_analysis(df):
    """고급 분석 표시"""
    st.subheader("🧠 고급 분석")
    
    # 현재 사용자의 데이터만 필터링
    user_df = df[df['user_id'] == st.session_state.user_id]
    
    if user_df.empty:
        st.info("분석할 데이터가 없습니다. 먼저 진단을 완료해주세요.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 MBTI 상관관계 분석")
        if len(user_df) > 1:
            fig, corr = create_correlation_heatmap(user_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상관관계 분석을 위해서는 최소 2개의 진단 데이터가 필요합니다.")
    
    with col2:
        st.subheader("🌐 MBTI 네트워크 분석")
        if len(user_df) > 1:
            fig, stats = create_mbti_network(user_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # 네트워크 통계 표시
                st.subheader("📈 네트워크 통계")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("노드 수", stats['total_nodes'])
                with col2:
                    st.metric("연결 수", stats['total_edges'])
                with col3:
                    st.metric("밀도", f"{stats['density']:.3f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 클러스터링", f"{stats['avg_clustering']:.3f}")
                with col2:
                    st.metric("연결 요소", stats['connected_components'])
            else:
                st.info(stats)  # stats가 문자열인 경우 (에러 메시지)
        else:
            st.info("네트워크 분석을 위해서는 최소 2개의 진단 데이터가 필요합니다.")
    
    # 추가 분석
    st.subheader("🔍 심화 분석")
    
    if len(user_df) > 1:
        # MBTI 변화 패턴 분석
        st.write("**🔄 MBTI 변화 패턴**")
        user_df_sorted = user_df.sort_values('timestamp')
        changes = []
        for i in range(1, len(user_df_sorted)):
            prev_mbti = user_df_sorted.iloc[i-1]['mbti']
            curr_mbti = user_df_sorted.iloc[i]['mbti']
            if prev_mbti != curr_mbti:
                changes.append({
                    'from': prev_mbti,
                    'to': curr_mbti,
                    'date': user_df_sorted.iloc[i]['timestamp']
                })
        
        if changes:
            st.info(f"총 {len(changes)}번의 MBTI 변화가 있었습니다.")
            change_df = pd.DataFrame(changes)
            st.dataframe(change_df, use_container_width=True)
        else:
            st.success("MBTI 유형이 일관되게 유지되고 있습니다.")
        
        # 시간대별 분석
        st.write("**⏰ 시간대별 분석**")
        user_df['hour'] = pd.to_datetime(user_df['timestamp']).dt.hour
        hour_counts = user_df['hour'].value_counts().sort_index()
        
        fig_hour = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            title="시간대별 진단 분포",
            labels={'x': '시간', 'y': '진단 수'}
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("심화 분석을 위해서는 최소 2개의 진단 데이터가 필요합니다.")

def show_data_management(df):
    """데이터 관리 표시"""
    st.subheader("📋 데이터 관리")
    
    # 현재 사용자의 모든 로봇 MBTI 이력
    with st.expander("내 모든 로봇 MBTI 진단/변화 이력", expanded=True):
        my_records = df[df['user_id']==st.session_state.user_id].sort_values('timestamp')
        if not my_records.empty:
            st.dataframe(my_records[["timestamp", "robot_id", "mbti", "gender", "age_group", "job"]], 
                       use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 진단 수", len(my_records))
            with col2:
                st.metric("사용한 로봇 수", my_records['robot_id'].nunique())
            with col3:
                st.metric("MBTI 유형 수", my_records['mbti'].nunique())
        else:
            st.info("아직 진단 이력이 없습니다.")
    
    # 데이터 다운로드 (현재 사용자 데이터만)
    st.subheader("📥 데이터 다운로드")
    user_df = df[df['user_id'] == st.session_state.user_id]
    
    if not user_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("내 데이터 CSV", 
                             user_df.to_csv(index=False).encode("utf-8"), 
                             f"{st.session_state.user_id}_data.csv", "text/csv")
        with col2:
            st.download_button("내 데이터 JSON", 
                             user_df.to_json(orient="records", force_ascii=False).encode("utf-8"), 
                             f"{st.session_state.user_id}_data.json", "application/json")
    else:
        st.info("다운로드할 데이터가 없습니다.")

def show_admin_data_management(df):
    """관리자 전용 데이터 관리"""
    st.subheader("🔧 관리자 데이터 관리")
    
    if not st.session_state.admin_logged_in:
        st.warning("관리자 로그인이 필요합니다.")
        return
    
    # 데이터 통계
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 진단 수", len(df))
    with col2:
        st.metric("고유 사용자 수", df['user_id'].nunique())
    with col3:
        st.metric("고유 로봇 수", df['robot_id'].nunique())
    with col4:
        st.metric("MBTI 유형 수", df['mbti'].nunique())
    
    # 데이터 관리 탭
    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs([
        "📊 전체 데이터", "🗑️ 중복 데이터 정리", "📥 데이터 내보내기", "⚙️ 시스템 관리"
    ])
    
    with admin_tab1:
        st.subheader("📊 전체 진단 데이터")
        
        # 필터링 옵션
        col1, col2, col3 = st.columns(3)
        with col1:
            user_filter = st.text_input("사용자 ID 필터", placeholder="특정 사용자 검색")
        with col2:
            robot_filter = st.text_input("로봇 ID 필터", placeholder="특정 로봇 검색")
        with col3:
            mbti_filter = st.selectbox("MBTI 유형 필터", ["전체"] + list(df['mbti'].unique()))
        
        # 필터링 적용
        filtered_df = df.copy()
        if user_filter:
            filtered_df = filtered_df[filtered_df['user_id'].str.contains(user_filter, case=False, na=False)]
        if robot_filter:
            filtered_df = filtered_df[filtered_df['robot_id'].str.contains(robot_filter, case=False, na=False)]
        if mbti_filter != "전체":
            filtered_df = filtered_df[filtered_df['mbti'] == mbti_filter]
        
        st.dataframe(filtered_df, use_container_width=True)
        st.info(f"필터링된 결과: {len(filtered_df)}건")
    
    with admin_tab2:
        st.subheader("🗑️ 중복 데이터 정리")
        
        # 중복 진단 확인
        duplicates = df.groupby(['user_id', 'robot_id']).size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]
        
        if not duplicates.empty:
            st.warning(f"중복 진단 발견: {len(duplicates)}개 사용자-로봇 조합")
            st.dataframe(duplicates, use_container_width=True)
            
            if st.button("중복 데이터 정리", type="primary"):
                # 중복 제거 (최신 데이터만 유지)
                df_cleaned = df.sort_values('timestamp').drop_duplicates(
                    subset=['user_id', 'robot_id'], 
                    keep='last'
                )
                
                # 데이터베이스 업데이트 (실제 구현에서는 더 안전한 방법 사용)
                st.info("중복 데이터 정리 기능은 데이터베이스 직접 수정이 필요합니다.")
                st.info(f"정리 전: {len(df)}건 → 정리 후: {len(df_cleaned)}건")
        else:
            st.success("중복 데이터가 없습니다.")
    
    with admin_tab3:
        st.subheader("📥 데이터 내보내기")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("전체 데이터 CSV", 
                             df.to_csv(index=False).encode("utf-8"), 
                             "all_diagnosis_data.csv", "text/csv")
        with col2:
            st.download_button("전체 데이터 JSON", 
                             df.to_json(orient="records", force_ascii=False).encode("utf-8"), 
                             "all_diagnosis_data.json", "application/json")
        
        # 통계 리포트 생성
        st.subheader("📈 통계 리포트")
        report_data = {
            "총 진단 수": len(df),
            "고유 사용자 수": df['user_id'].nunique(),
            "고유 로봇 수": df['robot_id'].nunique(),
            "가장 많은 MBTI": df['mbti'].mode().iloc[0] if not df['mbti'].mode().empty else "N/A",
            "평균 연령대": df['age_group'].mode().iloc[0] if not df['age_group'].mode().empty else "N/A",
            "성별 분포": df['gender'].value_counts().to_dict()
        }
        
        st.download_button("통계 리포트 JSON", 
                         str(report_data).encode("utf-8"), 
                         "diagnosis_report.json", "application/json")
    
    with admin_tab4:
        st.subheader("⚙️ 시스템 관리")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("데이터베이스 상태")
            try:
                # 간단한 연결 테스트
                test_result = supabase.table("responses").select("id").limit(1).execute()
                st.success("✅ 데이터베이스 연결 정상")
                st.info(f"총 레코드 수: {len(df)}")
            except Exception as e:
                st.error(f"❌ 데이터베이스 오류: {e}")
        
        with col2:
            st.subheader("시스템 정보")
            st.info(f"현재 사용자: {st.session_state.user_id}")
            st.info(f"관리자 로그인: {'예' if st.session_state.admin_logged_in else '아니오'}")
            st.info(f"현재 페이지: {st.session_state.page}")

# 메인 실행
if __name__ == "__main__":
    show_sidebar()
    show_main_content()