import cv2
import glob
from imutils import face_utils
import numpy as np
import imutils
import dlib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

LANDMARKS_COUNT = 68

#######################################################################################
##############                   Math and transfornatioms                  ############
#######################################################################################
def squared_distance(x,y):
    return (x[0]-y[0])**2+(x[1]-y[1])**2

def rect_to_bb(rect):
    """ take a bounding predicted by dlib and convert it
     to the format (x, y, w, h) as we would normally do
     with OpenCV
     *from web tutorial*"""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h) # return a tuple of (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    """
    *from web tutorial*
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def nparray_to_pandas_images(faces_68_landmarks):
    """
    input - nparray of numpy array (list of images numpy array, which contains 68 cords(tuple))
    output - pandas dataframe of data
    """
    df = pd.DataFrame.from_records(faces_68_landmarks)
    return df
    
#######################################################################################
##############                   Point Methods                             ############
#######################################################################################

def dot_matrix(point_arr):
    """
    input - nparray of (x, y) points
    output - an nxn matrix M where M[i, j] is the dot product of (xi, yi) and (xj, yj)
    """
    dot_m = np.ndarray(shape=(len(point_arr), len(point_arr)), dtype=int)
    for i in range(len(point_arr)):
        for j in range(i+1):
            dot_m[i, j] = np.dot(point_arr[i], point_arr[j])
            dot_m[j, i] = dot_m[i, j]
    return dot_m

def dist_matrix(dot_m):
    """
    input - a dot matrix (output of dot_matrix method)
    output - an nxn matrix M where M[i, j] is the distance between (xi, yi) and (xj, yj)
    """
    dist_m = np.ndarray(shape=dot_m.shape, dtype=float)
    for i in range(dist_m.shape[0]):
        dist_m[i, i] = 0
        for j in range(i):
            dist_m[i, j] = np.sqrt(dot_m[i, i] - 2*dot_m[i, j] + dot_m[j, j])
            dist_m[j, i] = dist_m[i, j]
    return dist_m

def angle_array(dot_m, dist_m):
    angles = []
    for i in range(angle_m.shape[0]):
        for j in range(i):
            for k in range(j):
                angles.append(np.arccos(
                    (dot_m[i, k] - dot_m[i, j] - dot_m[j, k] + dot_m[j, j]) / (dist_m[i, j] * dist_m[j, k])))
                angles.append(np.arccos(
                    (dot_m[i, j] - dot_m[i, k] - dot_m[k, j] + dot_m[k, k]) / (dist_m[i, k] * dist_m[k, j])))
                angles.append(np.pi - angles[-1] - angles[-2])
    return np.array(angles)


#######################################################################################
##############            Detecting face and face landmarks                ############
#######################################################################################
    
def detect_faces_CascadeClassifier(inputFolder,outputFolder):
    """
    foreach image in inputFolder: convert to gray scale, search faces with cv2.CascadeClassifier, cut, resize and save face in outputFolder
    *from web tutorial*
    *not used*
    """
    files = glob.glob("%s\\*"%inputFolder) #Get list of all images in inputFolder
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print ("face found in file: ", f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("%s\\%s.jpg" %(outputFolder, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

def extract_dlib_facial_points(inputFolder):
    """
    input - images folder name
    output - ndarray of images facial landmarks 
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    files = glob.glob("%s\\*"%inputFolder) #Get list of all images in inputFolder
    faces_68_landmarks = []
    for f in files:
        if f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"): 
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(f)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)
            # loop over the face detections
            for (i, rect) in enumerate(rects): #commented cause we want 1 face per image
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                faces_68_landmarks.append(shape)
    return np.array(faces_68_landmarks)
                

#######################################################################################
##############            Extract features and reducing dimensions         ############
#######################################################################################

def get_correlated_cols(df, threshold=0.8):
    corr = df.corr()
    return [corr.columns[i] for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i, j]) >= threshold]

def extract_features(image):
    """
    input - nparray of facial landmarks (point (x,y))
    output - nparray of features per image
    """
    #distance features
    dot_m = dot_matrix(image)
    dist_m = dist_matrix(dot_m)
    #angles features
    angles = angle_array(dot_m, dist_m)
    #flatten and concat
    return np.append(dist_m.flatten(), angles)
    
def extract_features_forall(images):
    """
    input - ndarray of images facial landmarks (for each image a 68 long nparry of points)
    output - dataframe of images features
    """
    features = []
    for image in images:
        features.append(extract_features(image))
    cols = ["dist_{0:d}_{1:d}".format(i, j) for i in range(LANDMARKS_COUNT) for j in range(LANDMARKS_COUNT)]
    for i in range(LANDMARKS_COUNT):
        for j in range(i):
            for k in range(j):
                cols.append("angle_{2:d}_{1:d}_{0:d}".format(i, j, k))
                cols.append("angle_{1:d}_{2:d}_{0:d}".format(i, j, k))
                cols.append("angle_{2:d}_{0:d}_{1:d}".format(i, j, k))
    drop_cols = ["dist_{0:d}_{0:d}".format(i) for i in range(d)] + ["dist_{0:d}_{1:d}".format(i, j) for i in range(LANDMARKS_COUNT) for j in range(i)]
    df = pd.DataFrame(features, columns=cols).drop(drop_cols, axis=1)
    df = (df - df.mean()) #/ df.std()
    corr_cols = get_correlated_cols(df)
    df.drop(corr_cols)
    return df
    
def dimension_reduction_pca(df, components = 100):
    """
    input - dataframe of features & wanted dimension of features
    output - df with #components features per image.
    uses PCA from skylearn
    """
    #casting df to contain numbers
    #df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    #print("Is df contains Nan: ", np.isnan(df).any())
    X = np.asanyarray(df)
    print("is finit:", (np.isfinite(X)).all())
    #Standardize the Data
    features = list(df.columns.values)
    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    #dim reduction
    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    return principalDf
    
    
#######################################################################################
##############            TESTS                                            ############
#######################################################################################
        

def test_image_1():
    """
    outdated
    """
    inputFolder = "./images/"
    outputFolder = "./out/"
    images_landmarks = extract_dlib_facial_points(inputFolder)
    print("landmarks shape: ", str(images_landmarks.shape))
    #f = nparray_to_pandas_images(images_landmarks)
    features = extract_features(images_landmarks)
    print("features shape: %s ", str(features.shape))

def test_image_features():
    image1 =[(0,0),(0,1),(1,0),(1,1)]
    image2 =[(0,0),(0,1),(1,0),(2,2)]
    image3 =[(0,0),(1,1),(1,2),(3,1)]
    images = []
    images.append(image1)
    images.append(image2)
    images.append(image3)
    features = extract_features_forall(images)
    print("features shape: ", str(features.shape))
    print(features)
    df = dimension_reduction_pca(features, 10)
    print("features after PCA shape: ", str(df.shape))
    print(df)
#ip = imagePreprocessing()

test_image_features()