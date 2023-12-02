from datetime import datetime


class TimePredictor:
    def __init__(self, total_steps: int):
        self.start_time = datetime.now()
        self.total_steps = total_steps

    def predict(self, current_step: int):
        end_time = datetime.now()
        difference = end_time - self.start_time
        time_per_step = difference / current_step
        steps_remaining = self.total_steps - current_step
        return f"{current_step} / {self.total_steps} | {time_per_step * steps_remaining} | {time_per_step} | {difference}"

    def get_total_duration(self):
        end_time = datetime.now()
        return end_time - self.start_time
