-- 11. Write a script that lists all shows contained in the database hbtn_0d_tvshows.
-- Each record should display: tv_shows.title - tv_show_genres.genre_id
-- Results sortered in ascending order by tv_shows.title and tv_show_genres.genre_id
-- If a show doesnt have a genre, display NULL
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command

-- SELECT COLUMNS TO DISPLAY FROM TABLE WHERE THE INFO IS LOCATED
   SELECT tv_genres.name AS genre,
   COUNT(tv_show_genres.genre_id) AS num_shows

-- WHERE THE FIELD IS EQUAL TO


-- FROM THE TABLE ORIGIN
        FROM tv_genres

-- JOIN CRITERIA
        INNER JOIN tv_show_genres
        ON tv_genres.id = tv_show_genres.genre_id

-- HOW THE DATA SHOULD BE ORGANIZED
        GROUP BY tv_show_genres.genre_id
        ORDER BY num_shows DESC;
