-- 진단 세션 ID 컬럼 추가
-- 이 스크립트는 기존 responses 테이블에 diagnosis_session_id 컬럼을 추가합니다.

-- 1. diagnosis_session_id 컬럼 추가
ALTER TABLE responses ADD COLUMN IF NOT EXISTS diagnosis_session_id TEXT;

-- 2. 기존 데이터에 대한 기본값 설정 (선택사항)
-- UPDATE responses SET diagnosis_session_id = CONCAT('legacy_', id) WHERE diagnosis_session_id IS NULL;

-- 3. 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_responses_diagnosis_session_id ON responses(diagnosis_session_id);
CREATE INDEX IF NOT EXISTS idx_responses_user_robot_session ON responses(user_id, robot_id, diagnosis_session_id);

-- 4. 중복 진단 세션 방지를 위한 유니크 제약 조건 (선택사항)
-- ALTER TABLE responses ADD CONSTRAINT unique_diagnosis_session UNIQUE (diagnosis_session_id);

-- 5. 테이블 구조 확인
-- \d responses; 