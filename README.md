# 🤖 로봇 MBTI 진단 시스템 (ProjHRI)

MBTI 기반 인간-로봇 상호작용(HRI) 진단 및 분석 시스템입니다. 사용자의 MBTI 유형을 분석하여 로봇과의 최적 상호작용 방식을 제안합니다.

## 📋 목차

- [주요 기능](#-주요-기능)
- [설치 및 실행](#-설치-및-실행)
- [사용 방법](#-사용-방법)
- [기능 상세](#-기능-상세)
- [데이터베이스 설정](#-데이터베이스-설정)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)

## ✨ 주요 기능

### 🧠 MBTI 진단 시스템
- **16가지 MBTI 유형 진단**: 정확한 MBTI 유형 분석
- **장소별 특화 설문**: 병원, 도서관, 쇼핑몰, 학교, 공항별 맞춤 질문
- **중복 진단 방지**: 24시간 내 중복 진단 자동 감지 및 방지
- **실시간 결과 분석**: 즉시 MBTI 결과 및 상호작용 가이드 제공

### 📊 데이터 분석 및 시각화
- **트렌드 분석**: 기간별 MBTI 분포 및 변화 추이
- **집단별 분석**: 성별, 연령대, 직업별 MBTI 분포 분석
- **로봇 이력 관리**: 개별 로봇별 진단 이력 및 변화 추적
- **MBTI 네트워크 분석**: MBTI 유형 간 관계 및 연결성 분석

### 🔧 관리자 기능
- **데이터 관리**: 전체 진단 데이터 조회 및 관리
- **중복 데이터 정리**: 중복 진단 데이터 자동 정리
- **통계 리포트**: 상세한 진단 통계 및 분석 리포트
- **데이터 내보내기**: CSV, JSON 형태로 데이터 내보내기

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 정보를 입력하세요:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 3. 데이터베이스 설정
Supabase SQL Editor에서 다음 스크립트를 실행하세요:

```sql
-- responses 테이블 업데이트
ALTER TABLE responses ADD COLUMN IF NOT EXISTS diagnosis_session_id TEXT;

-- user_robots 테이블 생성
CREATE TABLE IF NOT EXISTS user_robots (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    robot_name TEXT NOT NULL,
    robot_description TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 4. 애플리케이션 실행
```bash
streamlit run mbti_16_analysis_fixed.py
```

## 📖 사용 방법

### 1. 사용자 등록
- 사용자 ID 입력 (2-20자, 영문/한글/숫자)
- 성별, 연령대, 직업 정보 설정

### 2. 로봇 등록
- 로봇 ID 등록 (예: 내식기1, 로봇A)
- 여러 로봇 등록 가능

### 3. 진단 진행
- 진단 장소 선택 (일반, 병원, 도서관, 쇼핑몰, 학교, 공항)
- 장소별 특화 질문 응답 (총 17개 질문)
- MBTI 결과 확인 및 가이드 제공

### 4. 결과 분석
- 개인별 진단 이력 확인
- 로봇별 상호작용 패턴 분석
- MBTI 변화 추이 시각화

## 🔍 기능 상세

### 🏥 장소별 특화 설문

#### **병원 환경**
- 환자 정보 확인 및 신뢰성 검증
- 의료진과의 협력 상호작용
- 치료 과정 안내 및 모니터링
- 응급 상황 대응

#### **도서관 환경**
- 도서 검색 및 추천 시스템
- 학습 환경 조성
- 도서 대출/반납 서비스
- 이용 규칙 안내

#### **쇼핑몰 환경**
- 상품 추천 및 정보 제공
- 할인 정보 및 쇼핑 가이드
- 매장 위치 안내
- 고객 서비스 제공

#### **학교 환경**
- 수업 보조 및 학습 지원
- 학생 질문 응답
- 그룹 활동 관리
- 과제 및 일정 관리

#### **공항 환경**
- 수하물 및 보안 검사 안내
- 항공편 정보 제공
- 여행 계획 및 일정 관리
- 긴급 상황 대응

### 📈 분석 기능

#### **트렌드 분석**
- 기간별 MBTI 분포 시각화
- 라인/바/영역 차트 지원
- 빠른 기간 선택 (최근 7일, 30일, 3개월)
- MBTI별 상세 통계

#### **집단별 분석**
- 성별, 연령대, 직업별 MBTI 분포
- 바 차트, 파이 차트, 히트맵 지원
- 상세 통계 및 비율 분석

#### **로봇 이력 관리**
- 개별 로봇별 진단 이력
- MBTI 변화 타임라인
- 상세 진단 기록 테이블
- 변화 패턴 분석

#### **MBTI 네트워크 분석**
- MBTI 유형 간 관계 시각화
- 공통 특성 기반 연결성 분석
- 네트워크 통계 (밀도, 클러스터링)
- 심화 분석 (변화 패턴, 시간대별 분석)

### 🔧 관리자 기능

#### **데이터 관리**
- 전체 진단 데이터 조회
- 필터링 및 검색 기능
- 중복 데이터 정리
- 데이터베이스 상태 모니터링

#### **통계 및 리포트**
- 총 진단 수, 고유 사용자 수
- MBTI 유형별 분포 통계
- CSV, JSON 형태 데이터 내보내기
- 상세 통계 리포트 생성

## 🗄️ 데이터베이스 설정

### Supabase 설정
1. Supabase 프로젝트 생성
2. 환경 변수 설정
3. 테이블 생성 스크립트 실행

### 테이블 구조
- **responses**: 진단 응답 데이터
- **user_robots**: 사용자별 로봇 정보

## 🛠️ 기술 스택

### Frontend
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Plotly**: 인터랙티브 차트 및 시각화
- **Pandas**: 데이터 처리 및 분석

### Backend
- **Supabase**: 데이터베이스 및 인증
- **Python**: 백엔드 로직
- **NetworkX**: 네트워크 분석

### 데이터 분석
- **Scikit-learn**: 머신러닝 분석
- **SciPy**: 통계 분석
- **Seaborn**: 데이터 시각화

## 📁 프로젝트 구조

```
ProjHRI/
├── mbti_16_analysis_fixed.py    # 메인 애플리케이션
├── requirements.txt              # 의존성 패키지
├── .env                         # 환경 변수
├── README.md                    # 프로젝트 문서
├── create_user_robots_table.sql # 데이터베이스 스키마
├── update_database_schema.sql   # 스키마 업데이트
└── add_diagnosis_session_id.sql # 세션 ID 컬럼 추가
```

## 🔐 보안 기능

### 입력 검증
- 사용자 ID 및 로봇 ID 유효성 검사
- XSS 방지를 위한 입력 정제
- 특수문자 제한 및 길이 제한

### 데이터 보호
- 세션 기반 중복 진단 방지
- 진단 세션 ID를 통한 데이터 무결성 보장
- 관리자 권한 기반 데이터 접근 제어

## 📊 성능 최적화

### 데이터베이스 최적화
- 인덱스 생성으로 쿼리 성능 향상
- 중복 데이터 자동 정리
- 효율적인 데이터 구조 설계

### 시각화 최적화
- 대용량 데이터 처리 최적화
- 인터랙티브 차트 성능 개선
- 반응형 레이아웃 지원

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

**버전**: 2.0.0  
**최종 업데이트**: 2024년 12월  
**개발자**: ProjHRI Team
