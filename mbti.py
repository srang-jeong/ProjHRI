import streamlit as st
import pandas as pd
import random
from datetime import datetime
from supabase import create_client
import pytz
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = "Malgun Gothic"   # Mac이면 "AppleGothic", 리눅스는 "NanumGothic"
plt.rcParams['axes.unicode_minus'] = False

# --- Supabase 설정 ---
SUPABASE_URL = "https://qoxxrwhfripgvfsvbpwe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFveHhyd2hmcmlwZ3Zmc3ZicHdlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NjQ4OTYsImV4cCI6MjA2OTM0MDg5Nn0.UGX4jnrUDrmsMk_P5L_zB7gz6xLtowrlNCDuH4uoDls"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
st.set_page_config(page_title="MBTI HRI UX 진단툴", layout="wide", page_icon="🤖")

# --- 세션 상태 ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{random.randint(1000,9999)}"
USER_ID = st.session_state.user_id

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {"gender":"남", "age_group":"20대", "job":"학생"}
if 'robot_id' not in st.session_state:
    st.session_state.robot_id = "로봇A"
if 'robot_list' not in st.session_state:
    st.session_state.robot_list = ["로봇A"]

#### MBTI 16유형 HRI 안내 데이터
def load_mbti_guide():
    return {
        "ENFJ": {"description":"밝고 친근, 공감 리더십 안내. 필요 파악 솔루션 지향.",
            "examples":["안녕하세요! 무엇 도와드릴까요?","일정은 10시, 궁금한 점 편하게 알려주세요."]},
        "ENTJ": {"description":"목표 중심, 체계적 안내. 논리적 명확성 강조.",
            "examples":["목표 달성을 위해 이렇게 안내해 드립니다.","곧 다음 과제가 시작됩니다."]},
        "ENTP": {"description":"창의적, 다양한 옵션 제시. 자유로운 대화 선호.",
            "examples":["새로운 접근법을 준비해봤어요.","어떤 옵션이 더 흥미로우신가요?"]},
        "ENFP": {"description":"진심 어린 격려, 감정 반영 안내 선호.",
            "examples":["당신의 생각이 궁금해요!","새로운 제안 함께 해봐요."]},
        "ESFJ": {"description":"정중·세심, 모두 편안히 참여 안내.",
            "examples":["도움이 필요하시면 언제든 말씀해 주세요.","모두가 함께할 수 있게 안내해드립니다."]},
        "ESFP": {"description":"즉각적 지원, 유쾌한 인터랙션.",
            "examples":["즐겁게 시작해요!","필요하신 것 있으면 바로 말씀해 주세요."]},
        "ESTJ": {"description":"명확한 규칙, 체계적 안내.",
            "examples":["정해진 절차대로 안내해 드립니다.","규칙을 꼭 지켜주세요."]},
        "ESTP": {"description":"실용적·빠른 문제해결 안내.",
            "examples":["바로 시작하면 어떨까요?","즉시 실행하는 게 좋을 것 같아요."]},
        "INFJ": {"description":"깊은 공감, 세심한 배려 안내.",
            "examples":["당신의 마음을 이해합니다.","의미 있는 경험을 함께 해요."]},
        "INFP": {"description":"가치·감정 존중, 자기표현 안내.",
            "examples":["당신의 감정을 소중히 생각합니다.","진심을 가장 중요하게 여겨요."]},
        "INTJ": {"description":"미래지향·전략적 조언 안내.",
            "examples":["목표 달성을 위한 계획을 제안합니다.","장기적인 비전을 안내드릴게요."]},
        "INTP": {"description":"분석적 사고, 논리적 탐구 안내.",
            "examples":["새롭게 해석해볼까요?","함께 이유를 분석해봐요."]},
        "ISFJ": {"description":"조용한 배려, 실질적 지원 안내.",
            "examples":["필요하실 때 바로 도와드릴게요.","편하게 이용하실 수 있게 신경 쓸게요."]},
        "ISFP": {"description":"온화감성, 자유·편안함 안내.",
            "examples":["당신만의 방식을 존중합니다.","원하실 때 자유롭게 사용하세요."]},
        "ISTJ": {"description":"정확, 책임 강조·단계별 안내.",
            "examples":["정확하게 결과를 안내합니다.","규정·지침에 따라 진행합니다."]},
        "ISTP": {"description":"간단·실용, 요청시 직접 안내.",
            "examples":["필요하실 때 말씀만 하세요.","간결히 결과만 안내드려요."]},
    }
guide_data = load_mbti_guide()

#### HRI 모델 목록
def load_hri_models():
    try:
        res = supabase.table("hri_models").select("*").execute()
        return res.data or []
    except Exception as e:
        st.error(f"HRI 모델 불러오기 오류: {e}")
        return []
hri_models = load_hri_models()

# --- 사용자 입력/로봇 ID 직접 생성·선택 UI ---
with st.sidebar:
    st.header("사용자/로봇 정보")
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
    st.subheader("로봇 ID 관리")
    if "robot_list" not in st.session_state:
        robot_opts = ["로봇A"]
    else:
        robot_opts = list(st.session_state.robot_list)
    new_robot = st.text_input("새 로봇 별칭 직접 등록(예: 내식기1)", key="new_robot_id")
    if st.button("로봇 등록"):
        if new_robot.strip() and new_robot.strip() not in robot_opts:
            robot_opts.insert(0, new_robot.strip())
            st.success(f"'{new_robot.strip()}' 등록 완료")
    robot_id = st.selectbox("진단 대상 로봇(선택)", robot_opts, index=0 if st.session_state.robot_id not in robot_opts else robot_opts.index(st.session_state.robot_id))
    st.session_state.robot_list = robot_opts
    st.session_state.robot_id = robot_id

# --- 질문/타이브레이커 등 진단 로직은 이전과 동일 ---
core_questions = [
    {"id":"Q1","text":"로봇이 먼저 인사를 건넬 때 기분은? (친근 vs 부담)",
     "choices":["친근함","부담감"],"axes":("E","I")},
    {"id":"Q2","text":"안내 방식 선호: 세부 vs 비유?",
     "choices":["단계별 세부 안내","전체 맥락 비유 안내"],"axes":("S","N")},
    {"id":"Q3","text":"문제 해결 접근: 논리 vs 공감?",
     "choices":["논리적 해결","공감적 지원"],"axes":("T","F")},
    {"id":"Q4","text":"일정 알림: 반복 vs 유연?",
     "choices":["반복적 알림","유연한 즉흥 알림"],"axes":("J","P")},
    {"id":"Q5","text":"감정표현: 좋음 vs 불편?",
     "choices":["친근함 증가","과다 표현 불편"],"axes":("F","T")},
    {"id":"Q6","text":"안내 후 답변: 요약 vs 옵션?",
     "choices":["간결 요약","다양한 옵션"],"axes":("J","P")}
]
tie_questions = {
    "EI": {"axes":("E","I"), "text":"사교 모임 vs 혼자 독서? (외향 vs 내향)", "choices":["사교 모임","독서"]},
    "SN": {"axes":("S","N"), "text":"세부 정보 vs 전체 그림? (감각 vs 직관)", "choices":["세부 정보","전체 그림"]},
    "TF": {"axes":("T","F"), "text":"데이터 분석 vs 감정 공유? (사고 vs 감정)", "choices":["분석","공감"]},
    "JP": {"axes":("J","P"), "text":"계획 일정 vs 즉흥 일정? (판단 vs 인식)", "choices":["계획","즉흥"]}
}
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
            choice = st.radio(cfg['text'], tie_choices, index=0, key=f"tie_{axis}")
            if choice == "- 선택하세요 -":
                st.warning("추가 설문 문항 응답을 선택해야 진단이 완성됩니다.")
                st.stop()
            # 실제 선택한 값만 점수에 반영
            scores[a if choice == cfg['choices'][1] else b] += 1
    return scores
def predict_type(scores):
    return ''.join([
        'E' if scores['E'] >= scores['I'] else 'I',
        'S' if scores['S'] >= scores['N'] else 'N',
        'T' if scores['T'] >= scores['F'] else 'F',
        'J' if scores['J'] >= scores['P'] else 'P'
    ])
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
    for q in core_questions:
        responses[q['id']] = st.radio(q['text'], q['choices'], key=q['id'])
    st.session_state['responses'] = responses
    if st.button("▶️ 결과 보기"):
        if None in responses.values():
            st.warning("모든 문항에 답변하세요.")
        else:
            st.session_state.page = 2
            st.rerun()
######### 2. 결과 #############
elif page == 2:
    st.header(f"2️⃣ [{st.session_state.robot_id}] 진단 결과·피드백")
    responses = st.session_state.get('responses', {})
    profile = st.session_state.user_profile
    robot_id = st.session_state.robot_id
    scores = compute_scores(responses)
    if scores is None:
        st.warning("모든 문항에 답해주세요.")
    else:
        scores = resolve_ties(scores)
        mbti = predict_type(scores)

        # --- 여기 수정: 한 번만 저장 플래그 ---
        if not st.session_state.get('saved_result', False):
            save_response(USER_ID, responses, mbti, scores, profile, robot_id)
            st.session_state['saved_result'] = True   # 저장함 표시

        prev_record = load_last_mbti(USER_ID, robot_id)
        prev_mbti = prev_record['mbti'] if prev_record else None
        st.success(f"🔍 [{robot_id}] 예측 MBTI 유형: **{mbti}**")
        st.info(generate_adaptive_feedback(mbti, prev_mbti))
        guide = guide_data.get(mbti)
        if guide:
            st.subheader("💡 해당 유형 안내 가이드")
            st.write(guide['description'])
            with st.expander("📖 시나리오 예시"):
                for ex in guide['examples']:
                    st.write(f"- {ex}")
        st.download_button("💾 결과 CSV", pd.DataFrame([{'유형': mbti}]).to_csv(index=False), f"{mbti}.csv")
        if st.button("▶️ 통계/히스토리 대시보드 이동"):
            st.session_state.page = 3
            st.rerun()
######### 3. 통계/히스토리 #############
elif page == 3:
    st.header("3️⃣ 전체 통계 · 로봇 이력 · 집단분석(통합)")
    try:
        res = supabase.table("responses").select("*").execute()
        if not res.data or len(res.data)==0:
            st.info("아직 데이터가 없습니다.")
        else:
            df = pd.DataFrame(res.data)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            # --- 1) 기간/MBTI 분포 트렌드 ---
            st.subheader("기간별 MBTI 트렌드")
            min_date, max_date = df['date'].min(), df['date'].max()
            if min_date == max_date:
                st.info(f"데이터 날짜: {min_date}")
                df_period = df[df['date']==min_date]
            else:
                date_sel = st.slider("조회 기간", min_value=min_date, max_value=max_date, value=(min_date, max_date))
                df_period = df[(df['date']>=date_sel[0])&(df['date']<=date_sel[1])]
            if not df_period.empty:
                chart_data = df_period.groupby(['date','mbti']).size().unstack(fill_value=0)
                st.line_chart(chart_data)
            # --- 2) 집단·로봇별 MBTI 분포 ---
            st.subheader("성별/연령/직업/로봇별 MBTI 분포 (Bar/Pie/Pivot)")
            group_col = st.selectbox("분포 분석 기준",["gender","age_group","job","robot_id"])
            group_df = df.groupby([group_col, "mbti"]).size().unstack(fill_value=0)
            st.bar_chart(group_df)
            st.write("피벗테이블", pd.pivot_table(df, index=group_col, columns="mbti",
                                                aggfunc="size", fill_value=0))

            # 집단별 개별 Pie Chart 반복 시각화
            for cat in group_df.index:
                fig, ax = plt.subplots(figsize=(2, 2))     # 크기 조정
                group_df.loc[cat].plot.pie(autopct="%.1f%%", ax=ax, startangle=90,
                                        counterclock=False, colors=plt.cm.Set3.colors)
                ax.set_ylabel('')
                ax.set_title(f"{group_col}={cat} MBTI 분포", fontsize=10, fontweight="bold")
                st.pyplot(fig)

            st.write("피벗테이블", pd.pivot_table(df, index=group_col, columns="mbti", aggfunc="size", fill_value=0))
            st.markdown("#### [Pie Chart(집단별)]")

            # --- 3) 내 로봇 MBTI 변화 히스토리 ---
            st.subheader(f"'{st.session_state.robot_id}'의 MBTI 변화 히스토리")
            bot_records = df[(df['user_id']==USER_ID) & (df['robot_id']==st.session_state.robot_id)].sort_values("timestamp")
            mbti_types = sorted(df['mbti'].dropna().unique())
            mbti_map = {k: v for v, k in enumerate(mbti_types)}
            bot_records['timestamp_short'] = pd.to_datetime(bot_records['timestamp']).dt.strftime("%m-%d")

            if not bot_records.empty:
                bot_records['mbti_num'] = bot_records['mbti'].map(mbti_map)
                fig, ax = plt.subplots(figsize=(2,1))
                ax.plot(bot_records['timestamp_short'], bot_records['mbti_num'],
                        marker='o', color="tab:blue", linewidth=2, markersize=8)
                ax.set_yticks(list(mbti_map.values()))
                ax.set_yticklabels(list(mbti_map.keys()), fontsize=9)
                ax.set_xlabel("시점", fontsize=9)
                ax.set_ylabel("MBTI", fontsize=9)
                ax.set_title(f"'{st.session_state.robot_id}' MBTI 변화", fontsize=10, fontweight="bold")
                plt.xticks(rotation=45, fontsize=9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(bot_records[["timestamp", "mbti"]])
            else:
                st.info(f"로봇 '{st.session_state.robot_id}' 이력 없음")

            # --- 4) 나의 모든 로봇 MBTI 이력 표 ---
            with st.expander("내 모든 로봇 MBTI 진단/변화 이력"):
                my_records = df[df['user_id']==USER_ID].sort_values('timestamp')
                st.dataframe(my_records[["timestamp","robot_id","mbti","gender","age_group","job"]])
            # --- 5) 전체 DB 다운로드 ---
            st.download_button("전체 데이터 CSV", df.to_csv(index=False).encode("utf-8"), "all_responses.csv", "text/csv")
            st.download_button("전체 데이터 JSON", df.to_json(orient="records", force_ascii=False).encode("utf-8"), "all_responses.json", "application/json")
            # --- 6) HRI모델 정보 ---
            if hri_models:
                st.subheader("HRI 모델(시나리오) 목록")
                for m in hri_models:
                    st.markdown(f"**{m.get('name','모델명 없음')}**: {m.get('description','')}")
    except Exception as e:
        st.error(f"통계 대시보드 오류: {e}")

    if st.button("진단 첫화면으로 돌아가기"):
        st.session_state.page = 1
        st.rerun()
st.markdown("---")
st.info("MBTI 기반 로봇 성격 진단툴(로봇ID 직접입력, 그룹/로봇별 분석, 고도화 시각화)")