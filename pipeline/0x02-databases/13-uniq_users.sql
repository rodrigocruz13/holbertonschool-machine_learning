--  script that lists all shows from hbtn_0d_tvshows_rate by their rating.

-- Each record should display: tv_shows.title - rating sum
-- Results must be sorted in descending order by the rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command


-- Creates a table users following these requirements:

CREATE TABLE IF NOT EXISTS users (

-- id, integer, never null, auto increment and primary key
-- email, string (255 characters), never null and unique
-- name, string (255 characters)
-- If the table already exists, your script should not fail

        id INT NOT NULL AUTO_INCREMENT,
	PRIMARY KEY (id),
        email VARCHAR(255) NOT NULL UNIQUE,
	name VARCHAR(255)
        );

-- Your script can be executed on any database