# MBTI 16유형 전체 로봇 성격유형 진단 및 설계 가이드 제공 툴 (Streamlit)
# 실제 논문 구조(설문→진단→가이드/시나리오 제공→결과 저장/분석)와 일치
# 16유형별 설계 가이드/예시 모두 포함

import streamlit as st
import pandas as pd

st.set_page_config(page_title="로봇 성격유형 진단툴", layout="centered")
st.title("MBTI 16유형 기반 로봇 성격유형 진단 및 설계 가이드")
st.markdown(":robot_face: **모든 유형별 Human-Robot Interaction 설계 툴**")

# --- (1) 설문 입력 ---
st.header("1. 서비스 정보 입력")
service = st.selectbox("로봇 서비스 목적을 선택하세요", ["병원 안내", "도서관 안내", "교육 보조", "상담/케어", "쇼핑 도우미", "기타"])
user_type = st.selectbox("주요 대상 사용자(복수 선택 가능)", ["아동/학생", "성인", "고령자", "장애인", "불특정 다수"])

st.header("2. 상호작용 성향 입력 (MBTI 기반)")
ei = st.radio("로봇이 먼저 인사/대화를 거는 것이 편하십니까?", ["예(E): 먼저 적극적으로 대화", "아니오(I): 요청시만 대화"])
sn = st.radio("안내/설명 방식 선호", ["S: 단계적, 구체적 설명", "N: 전체 맥락, 비유 중심 설명"])
tf = st.radio("문제 상황 시 대화 선호", ["T: 논리/분석 중심", "F: 공감/격려 중심"])
jp = st.radio("일정/반복 공지 vs 유연한 즉흥 안내", ["J: 명확, 반복적 일정/공지", "P: 유연, 즉흥적 대화/변화 수용"])

# --- (2) 진단 결과 도출 ---
def get_mbti(ei, sn, tf, jp):
    ei_map = "E" if ei.startswith("예") else "I"
    sn_map = sn[0]
    tf_map = tf[0]
    jp_map = jp[0]
    return ei_map + sn_map + tf_map + jp_map

mbti_type = get_mbti(ei, sn, tf, jp)

# --- (3) 16유형별 설계 가이드/시나리오 ---
mbti_guide = {
    "ENFJ": {
        "설명": "밝고 친근한 인사, 공감, 칭찬, 명확한 일정 안내, 먼저 도움 제안",
        "예시": [
            "\"안녕하세요! 무엇을 도와드릴까요?\" (밝은 목소리, 눈맞춤, 먼저 인사)",
            "\"힘내세요! 오늘 일정은 10시 진료입니다.\" (격려/칭찬)",
            "\"불편한 점 있으시면 언제든 말씀해 주세요.\" (공감)"
        ]
    },
    "ENFP": {
        "설명": "유연하고 창의적 대화, 공감과 격려, 비유/상상 활용, 즉흥적 반응",
        "예시": [
            "\"오늘은 어떤 일로 오셨나요? 무슨 이야기도 환영입니다!\" (자유로운 대화)",
            "\"혹시 새로운 아이디어나 도움이 필요하신가요?\" (창의적 접근)",
            "\"지금 기분을 색깔로 표현한다면?\" (비유/상상 대화)"
        ]
    },
    "ENTJ": {
        "설명": "목표·계획 중시, 논리적 설명, 리더십 행동, 문제해결 지향",
        "예시": [
            "\"오늘의 목표를 말씀해 주시면 최적의 계획을 안내해 드리겠습니다.\" (계획 제시)",
            "\"문제 발생 시 빠르고 논리적으로 해결 방안을 제시합니다.\" (분석적)",
            "\"다음 단계로 바로 진행할까요?\" (리더십)"
        ]
    },
    "ENTP": {
        "설명": "창의·혁신 중심, 즉흥적 대화, 다양한 관점 제시, 논쟁/토론",
        "예시": [
            "\"여러 가지 방법을 생각해볼 수 있어요! 같이 아이디어를 내볼까요?\" (아이디어 제시)",
            "\"반대 의견도 환영이에요. 토론해 볼까요?\" (토론/논쟁)",
            "\"새로운 해결책이 필요하다면 언제든 말씀해 주세요.\" (유연성)"
        ]
    },
    "ESFJ": {
        "설명": "상냥한 배려, 세심한 공지/일정 관리, 친화력/조력 강조",
        "예시": [
            "\"예약하신 일정 미리 안내드립니다. 궁금하신 점 있으신가요?\" (사전 공지)",
            "\"어려움이 있으시면 제가 도와드릴게요!\" (도움/조력)",
            "\"감사합니다, 오늘도 좋은 하루 보내세요!\" (상냥한 마무리)"
        ]
    },
    "ESFP": {
        "설명": "즉각적 반응, 활기찬 행동, 실용적 도움 제공, 친근한 대화",
        "예시": [
            "\"필요하신 게 있으시면 바로 말씀해 주세요!\" (즉각 응답)",
            "\"새 상품 입고 안내드릴까요?\" (쇼핑 도우미)",
            "\"함께 찾아볼까요?\" (동행/도움 제안)"
        ]
    },
    "ESTJ": {
        "설명": "규칙/일정 강조, 현실적/단계별 안내, 책임감, 신속 피드백",
        "예시": [
            "\"지금부터 안내드릴 순서를 차례대로 따라가 주세요.\" (단계별 안내)",
            "\"정해진 규정과 절차에 따라 처리하겠습니다.\" (현실/책임감)",
            "\"요청하신 작업을 바로 처리했습니다.\" (신속성)"
        ]
    },
    "ESTP": {
        "설명": "즉흥적, 현실적 문제 해결, 직접적/간단한 설명, 즉각적 행동",
        "예시": [
            "\"바로 도와드릴게요! 필요하신 건 직접 보여드릴 수 있어요.\" (즉시행동)",
            "\"문제는 현장에서 직접 해결할 수 있습니다.\" (현실적)",
            "\"질문은 짧고 간단하게 말씀해 주세요.\" (간결성)"
        ]
    },
    "INFJ": {
        "설명": "공감과 경청, 비전/미래 안내, 조용한 배려, 깊이 있는 조언",
        "예시": [
            "\"필요하신 점을 말씀해 주시면 조용히 경청하겠습니다.\" (경청)",
            "\"앞으로의 방향이나 꿈이 있으신가요? 함께 고민해드릴 수 있습니다.\" (비전 안내)",
            "\"위로가 필요하실 때 언제든 말씀해 주세요.\" (공감/조언)"
        ]
    },
    "INFP": {
        "설명": "이해와 공감, 이상/가치 중심, 상상력/비유 대화, 개별화 서비스",
        "예시": [
            "\"당신의 이야기를 진심으로 듣고 싶어요.\" (이해/공감)",
            "\"자유롭게 의견을 표현해 주세요.\" (개별화)",
            "\"지금의 감정을 한 단어로 표현한다면 무엇일까요?\" (비유/상상)"
        ]
    },
    "INTJ": {
        "설명": "계획적, 미래지향, 논리/분석, 효율 극대화, 전략적 설명",
        "예시": [
            "\"목표를 말씀해 주시면, 단계별로 전략을 세워드리겠습니다.\" (전략)",
            "\"불필요한 단계는 최소화하겠습니다.\" (효율/분석)",
            "\"긴 안목에서 바라본 제안을 드릴 수 있습니다.\" (미래지향)"
        ]
    },
    "INTP": {
        "설명": "논리/이론 중심, 깊이 있는 분석, 독립적, 자유로운 토론/탐구",
        "예시": [
            "\"질문이 있으시면 언제든 제기해 주세요. 분석해서 답변드리겠습니다.\" (탐구/분석)",
            "\"정답이 하나가 아닐 수도 있습니다. 다양한 가능성을 검토해볼게요.\" (유연성)",
            "\"새로운 정보를 얻으면 바로 반영하겠습니다.\" (지식/적응)"
        ]
    },
    "ISFJ": {
        "설명": "배려/돌봄, 세심한 안내, 반복적 일정 관리, 안정감/신뢰 제공",
        "예시": [
            "\"불편하지 않으시도록 꼼꼼히 안내해드릴게요.\" (세심함)",
            "\"정해진 시간마다 일정을 안내드립니다.\" (반복/안정)",
            "\"늘 곁에서 도울 준비가 되어 있습니다.\" (돌봄)"
        ]
    },
    "ISFP": {
        "설명": "조용하고 부드러운 소통, 실용적 도움, 미적/감각적 대화, 비형식적 접근",
        "예시": [
            "\"천천히 말씀하셔도 괜찮아요. 필요한 부분을 함께 찾을게요.\" (조용/실용)",
            "\"주변 환경에 불편한 점은 없으신가요?\" (감각/배려)",
            "\"자연스럽게 편하게 대해 주세요.\" (비형식)"
        ]
    },
    "ISTJ": {
        "설명": "규범/원칙 준수, 명확/구체적 안내, 반복적 일정, 책임감 강조",
        "예시": [
            "\"정해진 규칙에 따라 절차를 진행하겠습니다.\" (원칙/책임)",
            "\"다음 일정을 다시 안내드립니다.\" (반복/안내)",
            "\"준비가 끝나면 바로 안내하겠습니다.\" (신속/명확)"
        ]
    },
    "ISTP": {
        "설명": "조용히 대기, 요청시만 간결 안내, 감정표현 최소화, 즉흥 문제해결",
        "예시": [
            "\"필요하실 때 호출해 주세요.\" (조용히 대기, 요청시만 응답)",
            "\"도서관 안내가 필요하신가요? 아래 버튼을 눌러 주세요.\" (간결한 안내)",
            "\"문제 상황 발생 시 즉시 분석, 신속 조치 안내\" (분석/즉흥)"
        ]
    },
    "ESTP": {
        "설명": "즉흥적, 현실적 문제 해결, 직접적/간단한 설명, 즉각적 행동",
        "예시": [
            "\"바로 도와드릴게요! 필요하신 건 직접 보여드릴 수 있어요.\" (즉시행동)",
            "\"문제는 현장에서 직접 해결할 수 있습니다.\" (현실적)",
            "\"질문은 짧고 간단하게 말씀해 주세요.\" (간결성)"
        ]
    },
}

st.header("3. 추천 로봇 성격유형 및 설계 가이드")
st.subheader(f"추천 유형: {mbti_type}")
if mbti_type in mbti_guide:
    st.success(mbti_guide[mbti_type]["설명"])
    with st.expander("[대화/행동 시나리오 예시 펼치기]"):
        for ex in mbti_guide[mbti_type]["예시"]:
            st.write(ex)
else:
    st.warning("해당 유형의 설계 가이드는 추후 업데이트 예정입니다.")

# --- (4) 결과 저장/데이터 관리 예시 ---
if st.button("결과 저장 및 다운로드 예시"):
    result_df = pd.DataFrame({
        "서비스": [service],
        "사용자": [user_type],
        "로봇 MBTI": [mbti_type],
        "설계 가이드": [mbti_guide.get(mbti_type, {}).get("설명", "-")]
    })
    st.dataframe(result_df)
    st.download_button("CSV로 다운로드", data=result_df.to_csv(index=False), file_name="robot_mbti_result.csv")

# --- (5) 시각적/통계 확장 예시 ---
st.header("4. 설문 통계/사용자 데이터 시각화 예시")
st.info("(여러 사용자의 입력 데이터가 누적되면, 유형별 분포/선호도/만족도 등 분석 그래프를 Streamlit에서 자동 생성 가능)")

st.caption("* 본 프로토타입 코드는 논문 설계 및 MBTI 기반 로봇 맞춤형 HRI 연구에 실제 응용 가능하도록 작성됨.")
