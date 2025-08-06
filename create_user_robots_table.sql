-- user_robots 테이블 생성 스크립트
-- 이 스크립트를 Supabase SQL Editor에서 실행하세요

-- 1. user_robots 테이블 생성
CREATE TABLE IF NOT EXISTS user_robots (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    robot_name TEXT NOT NULL,
    robot_description TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_user_robots_user_id ON user_robots(user_id);
CREATE INDEX IF NOT EXISTS idx_user_robots_robot_name ON user_robots(robot_name);
CREATE INDEX IF NOT EXISTS idx_user_robots_user_robot ON user_robots(user_id, robot_name);

-- 3. 중복 방지를 위한 유니크 제약 조건 (선택사항)
-- ALTER TABLE user_robots ADD CONSTRAINT unique_user_robot UNIQUE (user_id, robot_name);

-- 4. 테이블 구조 확인
-- SELECT column_name, data_type, is_nullable 
-- FROM information_schema.columns 
-- WHERE table_name = 'user_robots';

-- 5. 테이블 권한 설정 (필요한 경우)
-- GRANT ALL PRIVILEGES ON TABLE user_robots TO authenticated;
-- GRANT ALL PRIVILEGES ON TABLE user_robots TO anon;

-- 6. RLS (Row Level Security) 설정 (필요한 경우)
-- ALTER TABLE user_robots ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can view their own robots" ON user_robots FOR SELECT USING (auth.uid()::text = user_id);
-- CREATE POLICY "Users can insert their own robots" ON user_robots FOR INSERT WITH CHECK (auth.uid()::text = user_id);
-- CREATE POLICY "Users can update their own robots" ON user_robots FOR UPDATE USING (auth.uid()::text = user_id);
-- CREATE POLICY "Users can delete their own robots" ON user_robots FOR DELETE USING (auth.uid()::text = user_id); 