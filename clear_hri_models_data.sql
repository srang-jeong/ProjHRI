-- hri_models 테이블의 샘플 데이터 삭제

-- 모든 데이터 삭제
DELETE FROM public.hri_models;

-- 삭제 확인
SELECT 
    COUNT(*) as remaining_records,
    'hri_models 테이블의 모든 데이터가 삭제되었습니다.' as status
FROM public.hri_models; 