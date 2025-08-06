-- 데이터베이스 스키마 업데이트 스크립트
-- 이 스크립트를 Supabase SQL Editor에서 실행하세요

-- 1. diagnosis_session_id 컬럼 추가 (PostgreSQL)
ALTER TABLE responses ADD COLUMN IF NOT EXISTS diagnosis_session_id TEXT;

-- 2. 기존 데이터에 대한 기본값 설정
UPDATE responses 
SET diagnosis_session_id = CONCAT('legacy_', id::text) 
WHERE diagnosis_session_id IS NULL;

-- 3. 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_responses_diagnosis_session_id ON responses(diagnosis_session_id);
CREATE INDEX IF NOT EXISTS idx_responses_user_robot_session ON responses(user_id, robot_id, diagnosis_session_id);

-- 4. 테이블 구조 확인
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'responses';

-- 5. 실행 결과 확인
SELECT COUNT(*) as total_records, 
       COUNT(diagnosis_session_id) as records_with_session_id
FROM responses; 