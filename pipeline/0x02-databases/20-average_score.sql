-- creates a stored procedure ComputeAverageScoreForUser that computes and
-- store the average score for a student
-- Procedure ComputeAverageScoreForUser is taking 1 input:
--  * user_id, a users.id value (assume user_id is linked 2 an existing users)

DELIMITER ||

-- TRIGGUER BEGGINING
CREATE PROCEDURE ComputeAverageScoreForUser(IN new_user_id INT)

-- WHEN TO ACTIVATE

-- PROCEDURE ACTIONS
BEGIN

-- What to do
        UPDATE users
        SET average_score = (SELECT AVG(score)
      		             FROM corrections
		             WHERE new_user_id = corrections.user_id)
-- Criteria
        WHERE id = new_user_id;

END ||

DELIMITER ;