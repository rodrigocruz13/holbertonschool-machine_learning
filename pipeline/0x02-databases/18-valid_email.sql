-- script that creates a trigger that resets the attribute valid_email only
-- when the email has been changed.

DELIMITER ||

-- TRIGGUER BEGGINING
CREATE TRIGGER _UPDATE_EMAIL_

-- WHEN TO ACTIVATE
BEFORE UPDATE
ON users
FOR EACH ROW

-- TRIGGER ACTIONS
BEGIN

-- Criteria
        IF OLD.email <> NEW.email THEN
-- What to do
                SET NEW.valid_email = 0;
        END IF;

END ||

DELIMITER ;