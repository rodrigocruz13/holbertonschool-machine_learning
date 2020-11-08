-- creates a stored procedure ComputeAverageWeightedScoreForUser that computes
-- and store the average weighted score for a student.

-- Requirements
-- * Procedure ComputeAverageScoreForUser is taking 1 input:
-- * user_id, a users.id value (assume user_id is linked to an existing users)


DELIMITER ||

-- PROCEDURE BEGGINING

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN new_user_id INT)

BEGIN

-- What to do

    UPDATE users
    SET average_score = (
                        SELECT SUM(weight * score) / SUM(weight)

-- Criteria
                            FROM projects JOIN corrections
                            ON projects.id = corrections.project_id
		    	            WHERE new_user_id = corrections.user_id)
                        WHERE new_user_id = users.id;
END ||

DELIMITER ;
