SIDE_WIDTH = 400
INTRO_IMG = "http://drive.google.com/uc?export=view&id=1j_N8poUJqmJ1PJSf9Fe5Wn9q0OmP9qo9"

DATASETS = [
    "SQUARE", "TRIANGLE", "CIRCLE"
]
DIFFICULTY = [
    "Low", "Medium", "High", "Extreme"
]
SIZE = [
    f"{i}x{i}" for i in range(5, 21, 5)
]


BASIC_KEY = {
    "low_5x5": "1VTJ7EVm6GbFqJkOYoPtJc75gArEifO23",
    "medium_5x5": "17FEF_k2k-EXhGsen1KE8Xw3BAv0tXrmm",
    "high_5x5": "1j9g5DbixIRnHWaU72UCI92uE7VREbCTo",
    "extreme_5x5": "1v4os00tvhhygdy2Y0ahyEAKyzo3D18ep"
}
BASIC_GIF = {
    k:"http://drive.google.com/uc?export=view&id="+v for k, v in BASIC_KEY.items()
}

SOLUTION_KEY = {
    "low_5x5": "1utizOmExYHIP7u-LHyTii38DonJMebGw",
    "medium_5x5": "1FutTPn0fW_ndM1jfzwdFy6rvsk7hnI9Z",
    "high_5x5": "1UaGADpnlxgYPUNWbvnQbsHyvTYvIuUZH",
    "extreme_5x5": "11qEFlDMJh6WsH__8Z7hOl3wytzYD1BXB"
}
SOLUTION_GIF = {
    k:"http://drive.google.com/uc?export=view&id="+v for k, v in SOLUTION_KEY.items()
}


SOL_SQUARE_5_LOW = [
    (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (2,4), 
    (3,4), (4,4), (4,3), (4,2), (4,1), (4,0), (3,0), (2,0),
    (2,1), (2,2)
]
SOL_SQUARE_5_MEDIUM = [
    (0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3),
    (4,4), (3,4), (2,4), (1,4), (0,4), (0,3), (1,2)
]
SOL_SQUARE_5_HIGH = [
    (1,4), (0,4), (0,3), (0,2), (0,1), (0,0), (1,0), (2,0), 
    (3,0), (4,0), (4,1), (4,2), (4,3), (4,4), (3,4), (2,4),
    (2,3), (2,2)
]
SOL_SQUARE_5_EXTREME = [
    (0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (3,1),
    (2,2), (1,2), (2,3), (3,3), (4,3), (4,4), (3,4), (2,4),
    (1,4), (0,4)
]