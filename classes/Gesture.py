class Gesture():
    def __init__(self):
        pass
    
    def detectCount(self, frame):
        # count = cv2.detectCount(frame)
        return count

    def detectGesture(self, frame):
        count = self.detectCount(frame)
        action = {
            0: "sign_out",
            2: "sign_up"
            5: "sign_in",
        }
        return action.get(count, None)