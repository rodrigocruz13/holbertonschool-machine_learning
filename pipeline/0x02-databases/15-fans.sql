--  script that ranks country origins of bands, ordered by the number of
--  (non-unique) fans

-- Import this table dump: metal_bands.sql.zip
-- Column names must be: origin and nb_fans

-- SELECT COLUMNS TO DISPLAY FROM TABLE WHERE THE INFO IS LOCATED
        SELECT origin,
        SUM(fans) AS nb_fans

-- WHERE THE FIELD IS EQUAL TO


-- FROM THE TABLE ORIGIN
        FROM metal_bands

-- JOIN CRITERIA


-- HOW THE DATA SHOULD BE ORGANIZED
        GROUP BY origin
        ORDER BY nb_fans DESC;

-- Your script can be executed on any database