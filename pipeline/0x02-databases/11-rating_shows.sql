--  script that lists all shows from hbtn_0d_tvshows_rate by their rating.

-- Each record should display: tv_shows.title - rating sum
-- Results must be sorted in descending order by the rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command


-- SELECT COLUMNS TO DISPLAY FROM TABLE WHERE THE INFO IS LOCATED
   SELECT tv_shows.title,
   SUM(tv_show_ratings.rate) AS rating

-- WHERE THE FIELD IS EQUAL TO


-- FROM THE TABLE ORIGIN
        FROM tv_shows

-- JOIN CRITERIA
        INNER JOIN tv_show_ratings
        ON tv_show_ratings.show_id = tv_shows.id

-- HOW THE DATA SHOULD BE ORGANIZED
        GROUP BY tv_shows.title
        ORDER BY rating DESC;
