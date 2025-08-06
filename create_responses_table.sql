-- responses 테이블 생성
-- MBTI 진단 응답 데이터를 저장하는 테이블

CREATE TABLE IF NOT EXISTS public.responses (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    gender TEXT,
    age_group TEXT,
    job TEXT,
    robot_id TEXT NOT NULL,
    responses JSONB,
    mbti TEXT,
    scores JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RLS (Row Level Security) 설정
ALTER TABLE public.responses ENABLE ROW LEVEL SECURITY;

-- 정책 설정 (모든 사용자가 읽기/쓰기 가능)
CREATE POLICY "Enable all access for responses" ON public.responses
    FOR ALL USING (true);

-- 인덱스 생성 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_responses_user_id ON public.responses(user_id);
CREATE INDEX IF NOT EXISTS idx_responses_timestamp ON public.responses(timestamp);
CREATE INDEX IF NOT EXISTS idx_responses_mbti ON public.responses(mbti);

-- 테이블 생성 확인
SELECT 'responses 테이블이 생성되었습니다.' as status; 