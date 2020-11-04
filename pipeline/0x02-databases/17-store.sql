DELIMITER ||

-- TRIGGUER BEGGINING
CREATE TRIGGER _UPDATE_ORDER_

-- WHEN TO ACTIVATE
AFTER INSERT
ON orders FOR EACH ROW

-- TRIGGER ACTIONS
BEGIN

-- Where to act
        UPDATE items
-- What to do
        SET quantity = quantity - new.number
-- Criteria
        WHERE new.item_name = items.name;

END ||

DELIMITER ;