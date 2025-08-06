-- hri_models 테이블 생성
-- HRI 모델 정보를 저장하는 테이블

CREATE TABLE IF NOT EXISTS public.hri_models (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RLS (Row Level Security) 설정
ALTER TABLE public.hri_models ENABLE ROW LEVEL SECURITY;

-- 정책 설정 (모든 사용자가 읽기/쓰기 가능)
CREATE POLICY "Enable all access for hri_models" ON public.hri_models
    FOR ALL USING (true);

-- 샘플 HRI 모델 데이터 삽입
INSERT INTO public.hri_models (name, description) VALUES
('친근한 안내자', '따뜻하고 친근한 톤으로 사용자를 안내하는 모델'),
('효율적인 조수', '빠르고 정확한 정보 제공에 중점을 둔 모델'),
('창의적 파트너', '새로운 아이디어와 창의적 해결책을 제안하는 모델'),
('신뢰할 수 있는 전문가', '전문적이고 신뢰할 수 있는 조언을 제공하는 모델')
ON CONFLICT (name) DO NOTHING;

-- 테이블 생성 확인
SELECT 'hri_models 테이블이 생성되었습니다.' as status; 