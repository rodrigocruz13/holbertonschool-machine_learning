# 0x00. Lockboxes
## Specializations - Machine Learning

You have n number of locked boxes in front of you. Each box is numbered sequentially from 0 to n - 1 and each box may contain keys to the other boxes. Write a method that determines if all the boxes can be opened.

* Prototype: def canUnlockAll(boxes)
* boxes is a list of lists
* A key with the same number as a box opens that box
* You can assume all keys will be positive integers
* The first box (boxes[0]) is unlocked
* Return True if all boxes can be opened, else return False

## Analysis

### 1st sample. Result: True
![Sample 1](https://imgur.com/3S4IhGi.jpg)

### 2nd sample. Result: True
![Sample 2](https://i.imgur.com/D5i6zv3.jpg)

### 3th sample. Result: False
![Sample 3](https://i.imgur.com/jYXhUD4.jpg)
