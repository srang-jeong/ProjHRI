-- responses 테이블에 location 컬럼 추가
-- 이 스크립트를 Supabase SQL Editor에서 실행하세요

-- 1. location 컬럼 추가
ALTER TABLE public.responses ADD COLUMN IF NOT EXISTS location TEXT DEFAULT '일반';

-- 2. 기존 데이터에 기본값 설정
UPDATE public.responses 
SET location = '일반' 
WHERE location IS NULL;

-- 3. 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_responses_location ON public.responses(location);

-- 4. 테이블 구조 확인
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'responses' 
ORDER BY ordinal_position;

-- 5. 실행 결과 확인
SELECT COUNT(*) as total_records, 
       COUNT(location) as records_with_location,
       location,
       COUNT(*) as count_per_location
FROM public.responses 
GROUP BY location;