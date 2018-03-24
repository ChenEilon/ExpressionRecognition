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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import time


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

REF_POINTS = [4, 14, 18, 20, 22, 23, 25, 27, 28, 31, 32, 36, 37, 38, 40, 42, 43, 45, 46, 47, 49, 51, 52, 53, 61, 63, 65, 67]
EMOTION_NUM = 8
EMOTIONS = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

#######################################################################################
##############                   Math and transformations                  ############
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
    

def dataset_from_ck(inputFolderCKData):
    print("CK+ dataset preparation...")
    #create train_data and train_lbls
    train_lbls = []
    print("analyzing {0:s}...".format(EMOTIONS[0]))
    facial_landmarks_data = extract_dlib_facial_points(inputFolderCKData + "\\" + EMOTIONS[0])
    emotion_len = facial_landmarks_data.shape[0]
    for i in range(emotion_len):
        train_lbls.append(0)
    for e in range(1,len(EMOTIONS)):
        print("analyzing {0:s}...".format(EMOTIONS[e]))
        tmp = extract_dlib_facial_points(inputFolderCKData + "\\" + EMOTIONS[e])
        np.append(facial_landmarks_data, tmp)
        for i in range(facial_landmarks_data.shape[0]-emotion_len):
            train_lbls.append(e)
        emotion_len = facial_landmarks_data.shape[0]
    print("CK+ dataset ready!...")
    return (facial_landmarks_data, train_lbls)


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

def dist_array(dist_m):
    """
    input - a distance matrix (output of dist_matrix method)
    output - an array of all of the distances, w/o duplicates
    """
    dists = []
    for i in range(dist_m.shape[0]):
        for j in range(i):
            dists.append(dist_m[i, j])
    return np.array(dists)

def angle_array(dot_m, dist_m):
    """
    input - a dot matrix (output of dot_matrix method), a distance matrix (output of dist_matrix method)
    output - an array of all of the angles, w/o duplicates
    """
    angles = []
    for i in range(dot_m.shape[0]):
        for j in range(i):
            for k in range(j):
                #TODO change solution to devision by 0
                if not (dist_m[i, j] * dist_m[j, k] * dist_m[i, k]):
                    angles.append(-1)
                    angles.append(-1)
                    # angles.append(-1)
                else:
                    angles.append(np.arccos(round(
                        (dot_m[i, k] - dot_m[i, j] - dot_m[j, k] + dot_m[j, j]) / (dist_m[i, j] * dist_m[j, k]),
                        15)))
                    angles.append(np.arccos(round(
                        (dot_m[i, j] - dot_m[i, k] - dot_m[k, j] + dot_m[k, k]) / (dist_m[i, k] * dist_m[k, j]),
                        15)))
                    # angles.append(np.pi - angles[-1] - angles[-2])
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
    wanted_landmarks = [i-1 for i in REF_POINTS]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    files = glob.glob("%s\\*"%inputFolder) #Get list of all images in inputFolder
    faces_landmarks = []
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
                #faces_68_landmarks.append(shape)
                faces_landmarks.append(shape[wanted_landmarks])
    return np.array(faces_landmarks)
                

#######################################################################################
##############            Extract features and reducing dimensions         ############
#######################################################################################

def reduce_correlated_cols(df, threshold=0.95):
    """
    input - df &threshold (if 1>|correlation|>threshold then dimension is reduced
    output - reduced df
    """
    corr = df.corr()
    corr = corr * np.fromfunction(lambda i, j: i > j, corr.shape)
    corr_cols = (corr > threshold).sum(axis=1)
    corr_cols = corr_cols[corr_cols > 0].axes[0].tolist()
    ret = df.drop(corr_cols, axis=1)
    return ret

def extract_features(image):
    """
    input - nparray of facial landmarks (point (x,y))
    output - nparray of features per image
    """
    #distance features
    dot_m = dot_matrix(image)
    dist_m = dist_matrix(dot_m)
    dists = dist_array(dist_m)
    #angles features
    angles = angle_array(dot_m, dist_m)
    #flatten and concat
    features_vector = np.concatenate((dists, angles))
    return features_vector
    
def extract_features_forall(images):
    """
    input - ndarray of images facial landmarks (for each image a 68 long nparry of points)
    output - dataframe of images features
    """
    features = []
    for image in images:
        features.append(extract_features(image))
    cols = ["dist_{1:d}_{0:d}".format(REF_POINTS[i], REF_POINTS[j]) for i in range(len(REF_POINTS)) for j in range(i)]
    for i in range(len(REF_POINTS)):
        for j in range(i):
            for k in range(j):
                cols.append("angle_{2:d}_{1:d}_{0:d}".format(REF_POINTS[i], REF_POINTS[j], REF_POINTS[k]))
                cols.append("angle_{1:d}_{2:d}_{0:d}".format(REF_POINTS[i], REF_POINTS[j], REF_POINTS[k]))
                # cols.append("angle_{2:d}_{0:d}_{1:d}".format(REF_POINTS[i], REF_POINTS[j], REF_POINTS[k]))
    df = pd.DataFrame(features, columns=cols)
    return df
    
def dimension_reduction_pca(df, components = 100):
    """
    input - dataframe of features & wanted dimension of features
    output - trained PCA
    uses PCA from skylearn
    """
    #Standardize the Data
    features = list(df.columns.values)
    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    #dim reduction
    pca = PCA(components)
    print(pca)
    projected = pca.fit_transform(x)
    print(projected)
    #principalDf = pd.DataFrame(data = projected) #TODO disable and not return normally.
    return projected
    
    
#######################################################################################
##############            Machine learning algorithms                      ############
#######################################################################################

# logistic regression
def log_reg_classifier(imgs_features, imgs_lbls):
    """
    input - list of featurs list
    output - logistic regression classifier
    """
    clf = LogisticRegression(C = 1, penalty = 'l2') #TODO check best C
    return clf.fit(imgs_features, imgs_lbls)

# SVM
def svm_classifier(imgs_features, imgs_lbls):
    """
    input - list of featurs list
    output - svm classifier
    """
    # Create a classifier: a support vector classifier
    svm_classifier = svm.SVC(C = 1) #TODO check best C
    # training
    return svm_classifier.fit(imgs_features, imgs_lbls)
    
# KNN
def knn_classifier(imgs_features, imgs_lbls):
    """
    input - list of featurs list
    output - knn classifier
    """
    knn = KNeighborsClassifier(n_neighbors = EMOTION_NUM)
    return knn.fit(imgs_features, imgs_lbls) 
    
    

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
    """
    outdated
    """
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
    (pca, df) = dimension_reduction_pca(features, 200)
    print("features after PCA shape: ", str(df.shape))
    print(df)

def test_images_flow(inputFolder):
    #1. extract facial landmarks
    t1 = time.time()
    images_landmarks = extract_dlib_facial_points(inputFolder)
    print("landmarks shape: ", str(images_landmarks.shape))
    #2. extract features df
    t2 = time.time()
    df = extract_features_forall(images_landmarks)
    print("features shape: ", str(df.shape))
    #print(df)
    #3. using correlation matrix to reduce dimension
    t3 = time.time()
    corrDf = reduce_correlated_cols(df)
    print("corr df shape: ", str(corrDf.shape))
    #print("no correlation here")
    #print(df)
    #4. reduce dimension with PCA
    t4 = time.time()
    m_pca = dimension_reduction_pca(corrDf, 150)
    print("df after PCA shape: ", m_pca.shape)
    t5 = time.time()
    #print(newDf)
    #timing report:
    print("Timing Report:")
    print("Extract landmarks:" + str(t2-t1))
    print("Extract features:" + str(t3-t2))
    print("Extract correlation:" + str(t4-t3))
    print("Extract pca:" + str(t5-t4))

def test_ml_algos():
    X = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
    y = [0,0,1,1,2,2,3,3,3,3,3]
    m_knn = knn_classifier(X,y)
    m_svm = svm_classifier(X,y)
    m_lin_log = log_reg_classifier(X,y)
    p1 = [0.5]
    p2 = [9]
    if(m_knn.predict([p1])[0] == m_svm.predict([p1])[0] == m_lin_log.predict([p1])[0] == 0):
        if(m_knn.predict([p2])[0] == m_svm.predict([p2])[0] == m_lin_log.predict([p2])[0] == 3):
            print("all good!")
            return
    print("One of the predictor failed! most likely lin log.. we are sad. we want to sleep. or go to the Thailand")
<<<<<<< HEAD
    
def test_ml_algos_on_ck(inputFolderCKData):
    print("Start testing...")
    (facial_landmarks_data, train_lbls) = dataset_from_ck(inputFolderCKData)
    features_df = extract_features_forall(facial_landmarks_data)
    #reduce dimensions
    print("Dim reduction...")
    pca = dimension_reduction_pca(features_df, 100)
    features_red = pca.transform(features_df)
    #training ml algos
    print("ml algos training...")
    m_knn = knn_classifier(features_red,train_lbls)
    m_svm = svm_classifier(features_red,train_lbls)
    m_lin_log = log_reg_classifier(features_red,train_lbls)
    #test ml algos
    print("KNN -  score on training data: ", m_knn.score(features_red))
    print("SVM -  score on training data: ", m_svm.score(features_red))
    print("Linear logistic - score on training data: ", m_lin_log.score(features_red))

    
        
=======
>>>>>>> 09221d616b5651f7a8d6eadc1fd9136c8b30482a

#######################################################################################
##############            RUN                                              ############
#######################################################################################
    
#test_image_features()
#test_images_flow(r"C:\Users\DELL1\Documents\studies\FinalProject\facial-landmarks\facial-landmarks\images")
#test_ml_algos()
<<<<<<< HEAD
test_ml_algos_on_ck(r"C:\Users\DELL1\Documents\studies\FinalProject\Datatsets\CK+\sorted_set")
=======
>>>>>>> 09221d616b5651f7a8d6eadc1fd9136c8b30482a
