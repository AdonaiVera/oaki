import numpy as np
from methods.body_part_angle import BodyPartAngle
from methods.utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def squat(self, counter, status):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

        if status:
            if avg_leg_angle < 70:
                counter += 1
                status = False
        else:
            if avg_leg_angle > 160:
                status = True

        return [counter, status]


    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "squat":
            counter, status = TypeOfExercise(self.landmarks).squat(
                counter, status)

        return [counter, status]
