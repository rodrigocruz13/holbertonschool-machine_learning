--  script that ranks country origins of bands, ordered by the number of
--  (non-unique) fans

-- Import this table dump: metal_bands.sql.zip
-- Column names must be: origin and nb_fans

-- SELECT COLUMNS TO DISPLAY FROM TABLE WHERE THE INFO IS LOCATED
        SELECT band_name,

-- CONDITIONS TO MEET
        IF(split IS NULL, (2020 - formed), (split - formed)) AS lifespan

-- FROM THE TABLE ORIGIN
        FROM metal_bands

-- JOIN CRITERIA


-- WHERE THE FIELD IS EQUAL TO
        WHERE style REGEXP "Glam rock"


-- HOW THE DATA SHOULD BE ORGANIZED
        ORDER BY lifespan DESC;

-- Your script can be executed on any database