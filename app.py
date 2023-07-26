'''TASK:Create a web application that can detect and count the number of vehicles passing through a video, using Opencv and Flask. 
Detection will be based on size of vehicles so that it can tell the vehicle is car or truck.'''
import cv2 
import numpy as np 
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
'''cv2 for computer vision, numpy for numerical operations, 
and Flask for web application development. The capture object is created to read frames from a video file named video.mp4.'''
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads/"
@app.route('/') 
def index():        
    return render_template('index.html') #rendering index.html where the action is set to upload to handle the uploads using
#a function created as uploads and where method is set to post 

@app.route('/upload', methods=['POST'])#as defined in index the route has been given in decorator and method as POST 
def upload():
    video = request.files['video'] #as defined in index the video variable is taking the uploaded video 
    #and retrieves the uploaded file named "video"
    
    if video.filename != '':#line 22 -24 main motive here is to check if the user has actually  uploaded the video and 
        #the code should work on the most recently uploaded video by removing the old video uploaded in past 
        for filename in os.listdir('static/uploads/'):
            os.remove('static/uploads/' + filename)
        filename = secure_filename(video.filename)#secure filename for further operations in code so there isn't any 
        #characters or file paths that could be used to compromise the server.
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        message = 'Video uploaded successfully'
        #making a capture object 
        capture=cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #count_line_position 
        count_line_position =320#sets a line position at 550 pixels, and an offset of 6 pixels to account for any small errors
        offset=6  #allowable error between pixels
        counter=0
        global car_counter,truck_counter #declared these variables global so they can accessed by flask as well 
        car_counter = 0
        truck_counter = 0
        #background subtractor to detect the moving objects in the video
        bgsb=cv2.createBackgroundSubtractorMOG2()

        min_bredth_rect=80 # declaring minimum widths and height of bounding rectangle
        min_height_rect=80

        def center_handle(x,y,b,h): #function to find the center of the moving object and then if it crosses the threshold(count line)
            # then only will be counted 
            x1=int(b/2)
            y1=int(h/2)
            cx=x+x1
            cy=y+y1
            return cx,cy
        detect=[]
        #infinity loop for continous reading of each frame with a break condition if any frame is not returned 
        while True: #or the loop will break only when the key z is pressed as per the waitkey condition 
            ret,frame=capture.read() #for reading the video ret will present boolean value i.e,frame was successfully read or not
            if not ret:                #frame returns a numpy array which contains the image data for the frame.   
                break
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#original color space to grayscale 
            blur=cv2.GaussianBlur(grey,(3,3),5)#Gaussian blur to the grayscale image 
            #(grayscale image,kernel size,standard deviation of the Gaussian kernel)
            #applying on all frame
            img_sub=bgsb.apply(blur)#background subtraction algorithm to the blurred image
            dilat=cv2.dilate(img_sub,np.ones((5,5)))#morphological dilation on the binary image (binary image,kernel for dilation)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#kernel for morphological closing(type of kernel,size of the kernel)
            dilakern=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)# morphological closing on the dilated image
            #(dilated image,type of morphological operation, kernel )
            dilakern=cv2.morphologyEx(dilakern,cv2.MORPH_CLOSE,kernel)#morphological closing again to fill the gaps in runtime
            #making the contours of object 
            contourss,h=cv2.findContours(dilakern,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#find contours of objects in the dilated image (dilakern)
            #contourss - a list of contours, and h - a hierarchy of contours
            cv2.line(frame,(200,count_line_position),(900,count_line_position),(122,100,255),3)
            #draws a line on the frame image to mark the virtual line on the road and the rest is positioning ,formatting of line
            for (i,j) in enumerate(contourss):#loops over contours returned by cv2.findContours()
                (x,y,b,h)=cv2.boundingRect(j)
                val_counter=(b>=min_bredth_rect) and (h>=min_height_rect)#removing small contours which may not be vehicles 
                if not val_counter:#if contour is not passing above condition loop will skip to next contour and continue 
                    continue
                cv2.rectangle(frame,(x,y),(x+b,y+h),(0,255,0),2)#if contour is passed rectangle will be drawn 
                center=center_handle(x,y,b,h)# center of the rectangle
                detect.append(center)   #adding the center coordinates to list 
                cv2.circle(frame,center,4,(0,0,255),-1)#circle of radius of 4 pixels is created at the center coordinates of the rectangle

                for (x,y) in detect:#if any point in detect list crosses the line from detect list 
                    if y< (count_line_position + offset) and y > (count_line_position - offset):
                        #center of the detected object is above the count line position, plus a given offset value
                        if b < 150: #if breadth is less than150 car counter is +1
                            car_counter += 1
                        else:           #if breadth is greater than 150 truck counter is +1
                            truck_counter += 1
                        cv2.line(frame, (200, count_line_position), (900, count_line_position), (0, 127, 255), 3)
                        detect.remove((x, y))#removes the detected point from detect list and print the current counts to console  
                        print("Cars:", car_counter, "Trucks:", truck_counter)
            
            #cv2.putText(frame, "Cars: " + str(car_counter) + " Trucks: " + str(truck_counter), (2,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            #adding text to the console ( image,text to display,position of the text,font type,font scale,color,thickness)
            #cv2.imshow('Detector',dilakern) #for checking the subtracted background and created contours 
            #to present the video 
            #cv2.imshow('Running Video',frame)
            #to terminate the loop
            #if cv2.waitKey(1) & 0xFF==ord('z'):#waitkey will wait in milliseconds for any input by user and if the key is pressed it'll break 
            #    break   #using & logical operator to compare both condns. 0xFF will make sure the value lies in the ASCII range of 0-255
        capture.release()
        # Flask route that returns an HTML template with the current car and truck counts.
        return render_template('result.html', car_count=car_counter, truck_count=truck_counter)
    else:
        message = 'No file selected'
        return render_template('index.html')
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response
    
if __name__ == '__main__':
    app.run(debug=True,port=5000,host="0.0.0.0")