class Gesture():
    def __init__(self, cascade_file):
      
        # Load Cascade Facial Detection
        self.detector = cv2.CascadeClassifier(cascade_file)
    
    def detectCount(self, frame):
        # count = cv2.detectCount(frame)
        copy = cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)
        kernel = np.ones((3,3),np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:
            # To draw a rectangle in a face 
            cv2.rectangle(copy,(x-30,y-50),(x+w+30,y+h+200),(0,0,0), -1)

        roi=copy[10:710, 10:1270]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin Range in HSV
        lower_skin = np.array([0,45,80], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
        # Extract all components that are within skin color range 
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Reduce noise by extrapolate hand image of dark spots 
        mask = cv2.dilate(mask,kernel,iterations = 3)

        # Calculate contours 
        _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Find the countour of max area 
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # Approx contour 
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

        # Convex hull of hand contour 
        hull = cv2.convexHull(cnt)

        # Define area around hull of hand 
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # Find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100

        # Calculate number of defects in hand 
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        # de = no. of defects
        de=0

        # Calculations of defects to fingers 
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            # Find length of triangles 
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            # Distance between convex point and hull
            d=(2*ar)/a
            
            # Consine rule for angle calculation 
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
            # Ignore angles that are > 90 and very close to convex hull
            if angle <= 90 and d>30:
                de += 1            
        de+=1

        # Assign value of fingers detected to count 
        font = cv2.FONT_HERSHEY_SIMPLEX
        if de==1:
            if areacnt<2000:
                count = NULL 
            else:
                if arearatio<12:
                    count = 0              
                else:
                    count = 1                    
        elif de==2:
            count = 2
        elif de==3:
            count = 3   
        elif de==4:
            count = 4 
        elif de==5:
            count = 5
        else :
            count = 10

        return count

    def detectGesture(self, frame):
        count = self.detectCount(frame)
        action = {
            0: "sign_out",
            2: "sign_up"
            5: "sign_in",
        }
        return action.get(count, None)