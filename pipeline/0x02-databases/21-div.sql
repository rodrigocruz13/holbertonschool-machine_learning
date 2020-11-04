-- script that creates a function SafeDiv that divides (and returns) the first
-- by the second number or returns 0 if the second number is equal to 0.

-- Req:
-- You must create a function
-- The function SafeDiv takes 2 arguments:
-- a, INT
-- b, INT
-- And returns a / b or 0 if b == 0

DELIMITER ||

-- FUNCTION BEGGINING
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT

BEGIN

-- Criteria
        RETURN (IF (b <> 0, a / b, 0));
END ||

DELIMITER ;