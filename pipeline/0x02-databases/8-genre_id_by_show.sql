-- 10. Import the DB dump from hbtn_0d_tvshows to your MySQL server: download
-- Write a script that lists all shows contained in hbtn_0d_tvshows that have 
-- at least one genre linked.
-- Each record should display: tv_shows.title - tv_show_genres.genre_id
-- Results must be sorted in asc order by tv_shows.title and tv_show_genres.genre_id
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command



-- SELECT COLUMNS TO DISPLAY FROM TABLE WHERE THE INFO IS LOCATED
   SELECT tv_shows.title, tv_show_genres.genre_id

-- WHERE THE FIELD IS EQUAL TO 


-- FROM THE TABLE ORIGIN
        FROM tv_shows INNER JOIN tv_show_genres

-- JOIN CRITERIA
        ON tv_shows.id = tv_show_genres.show_id

-- HOW THE DATA SHOULD BE ORGANIZED
        ORDER BY tv_shows.title, tv_show_genres.genre_id;

