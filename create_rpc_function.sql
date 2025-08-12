-- RPC 함수 생성: location 컬럼 자동 추가
-- 이 스크립트를 Supabase SQL Editor에서 실행하세요

-- 1. location 컬럼 추가 함수 생성
CREATE OR REPLACE FUNCTION add_location_column()
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- location 컬럼이 이미 존재하는지 확인
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'responses' 
        AND column_name = 'location'
    ) THEN
        -- location 컬럼 추가
        ALTER TABLE public.responses ADD COLUMN location TEXT DEFAULT '일반';
        
        -- 기존 데이터에 기본값 설정
        UPDATE public.responses 
        SET location = '일반' 
        WHERE location IS NULL;
        
        -- 인덱스 추가
        CREATE INDEX IF NOT EXISTS idx_responses_location ON public.responses(location);
        
        RETURN 'location 컬럼이 성공적으로 추가되었습니다.';
    ELSE
        RETURN 'location 컬럼이 이미 존재합니다.';
    END IF;
END;
$$;

-- 2. 함수 실행 권한 부여
GRANT EXECUTE ON FUNCTION add_location_column() TO anon;
GRANT EXECUTE ON FUNCTION add_location_column() TO authenticated;

-- 3. 함수 테스트
SELECT add_location_column();