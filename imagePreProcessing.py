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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

REF_POINTS = [4, 14, 18, 20, 22, 23, 25, 27, 28, 31, 32, 36, 37, 38, 40, 42, 43, 45, 46, 47, 49, 51, 52, 53, 61, 63, 65, 67]
EMOTIONS = ["neutral",  "happy", "sadness", "surprise",  "fear", "disgust", "anger", "contempt"] #Define emotions

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
    print("analyzing {0:s}...".format(EMOTIONS[0]))
    facial_landmarks_data = extract_dlib_facial_points(inputFolderCKData + "\\" + EMOTIONS[0])
    emotion_len = facial_landmarks_data.shape[0]
    train_lbls = [0 for i in range(emotion_len)]
    for e in range(1,len(EMOTIONS)):
        print("analyzing {0:s}...".format(EMOTIONS[e]))
        tmp = extract_dlib_facial_points(inputFolderCKData + "\\" + EMOTIONS[e])
        facial_landmarks_data = np.concatenate((facial_landmarks_data, tmp))
        train_lbls += [e for i in range(facial_landmarks_data.shape[0]-emotion_len)]
        emotion_len = facial_landmarks_data.shape[0]
    print("CK+ dataset ready!...")
    return (facial_landmarks_data, train_lbls)

def save_plt_scores(params, nameP, scores, nameScores, title, log_scale=True):
    fig = plt.figure()
    plt.grid(True)
    #axes = plt.gca()
    if log_scale:
        plt.semilogx()
    plt.plot(params, scores)
    plt.axis([min(params) , max(params) , min(scores) - 0.01, max(scores) + 0.01])
    plt.ylabel(nameScores)
    plt.xlabel(nameP)
    plt.title(title)
    #plt.show()
    fig.savefig(title+'.png')

def dataset_from_affectnet(trainingCsvPath):
    """
    problem with affectnet landmarks...
    """
    data_df = pd.read_csv(trainingCsvPath)
    df_filtered = data_df.query('expression<8')
    landmarks = (df_filtered[['facial_landmarks']].values).flatten()
    labels = df_filtered[['expression']].values
    facial_landmarks_data = [np.reshape(i.split(";"),(68,2)) for i in landmarks]
    return (facial_landmarks_data, labels.flatten())

    
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

def image_to_landmarks(image_path, detector, predictor):
    """assuming an image"""
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    if image is None:
        return []
    image = imutils.resize(image, width=350)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    if len(rects)==0:
        return []
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    #faces_68_landmarks.append(shape)
    return shape


        
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
            shape = image_to_landmarks(f, detector, predictor)
            faces_landmarks.append(shape[wanted_landmarks])
    return np.array(faces_landmarks)
                

def sort_sample_affectnet(inputFolder, csvPathAffectnet):
    """
    csv: 'image_name', 'expression', '68_landmarks'
    """
    #wanted_landmarks = [i-1 for i in REF_POINTS]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    data_df = pd.read_csv(csvPathAffectnet)
    landmarks = []
    folders = glob.glob(inputFolder + "\\*") #Returns a list of all folders with participant numbers
    for folder in folders:
        files = glob.glob(folder + "\\*")
        for f in files:
            shape = image_to_landmarks(f, detector, predictor)
            shape = shape.flatten()
            img_name = (f.split("\\"))[-1]
            tmp = [img_name] + list(shape)
            landmarks.append(tmp)
    landmarks_df = pd.DataFrame(landmarks, columns = ['image_name','landmarks'])
    data_df = data_df.join(landmarks_df.set_index('image_name'), on='image_name')
    data_df.to_csv('out.csv')
    

                
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
    pca.fit_transform(x)
    return pca
    
    
#######################################################################################
##############            Machine learning algorithms                      ############
#######################################################################################

# logistic regression
def log_reg_classifier(imgs_features, imgs_lbls, c=1):
    """
    input - list of featurs list
    output - logistic regression classifier
    """
    clf = LogisticRegression(C = c, penalty = 'l2') #TODO check best C
    return clf.fit(imgs_features, imgs_lbls)

# SVM
def svm_classifier(imgs_features, imgs_lbls, c=1):
    """
    input - list of featurs list
    output - svm classifier
    """
    # Create a classifier: a support vector classifier
    svm_classifier = svm.SVC(C = c) #TODO check best C
    # training
    return svm_classifier.fit(imgs_features, imgs_lbls)
    
# KNN
def knn_classifier(imgs_features, imgs_lbls, k=1):
    """
    input - list of featurs list
    output - knn classifier
    """
    knn = KNeighborsClassifier(n_neighbors = k) #TODO check best k
    return knn.fit(imgs_features, imgs_lbls) 
    
    

#######################################################################################
##############            TESTS                                            ############
#######################################################################################
        
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
    #4. reduce dimension with PCA
    t4 = time.time()
    m_pca = dimension_reduction_pca(corrDf, 150)
    t5 = time.time()
    #timing report:
    print("Timing Report:")
    print("Extract landmarks:" + str(t2-t1))
    print("Extract features:" + str(t3-t2))
    print("Extract correlation:" + str(t4-t3))
    print("Extract pca:" + str(t5-t4))
    return corrDf, m_pca

def test_ml_algos_on_ck(inputFolderCKData):
    print("Start testing...")
    (facial_landmarks_data, train_lbls) = dataset_from_ck(inputFolderCKData)
    features_df = extract_features_forall(facial_landmarks_data)
    #reduce dimensions
    print("Dim reduction...")
    pca = dimension_reduction_pca(features_df, 500)
    features_red = pca.transform(features_df)
    #training ml algos
    print("ml algos training...")
    m_knn = knn_classifier(features_red,train_lbls,3)
    m_svm = svm_classifier(features_red,train_lbls)
    m_log_reg = log_reg_classifier(features_red,train_lbls)
    #test ml algos
    print("KNN -  score on training data: ", m_knn.score(features_red,train_lbls))
    print("SVM -  score on traifeatures_df = extract_features_forall(facial_landmarks_data)ning data: ", m_svm.score(features_red, train_lbls))
    print("Logistic Regression - score on training data: ", m_log_reg.score(features_red,train_lbls))

def find_best_params(inputFolderCKData):
    scoresSVM = []
    scoresLogReg = []
    scoresKNN = []
    print("Start testing...")
    (facial_landmarks_data, facial_landmarks_lbls) = dataset_from_ck(inputFolderCKData)
    facial_landmarks_lbls = np.array(facial_landmarks_lbls)
    features_df = extract_features_forall(facial_landmarks_data)
    image_num = len(facial_landmarks_lbls)
    #reduce dimensions
    print("Dim reduction...")
    pca = dimension_reduction_pca(features_df, 1600)
    features_red = pca.transform(features_df)
    #dividing to train and validation
    randIndxs = list(range(image_num))
    random.shuffle(randIndxs)
    train_data = features_red[randIndxs[:image_num//2]]
    train_lbls = facial_landmarks_lbls[randIndxs[:image_num//2]]
    validation_data = features_red[randIndxs[image_num//2:]]
    validation_lbls = facial_landmarks_lbls[randIndxs[image_num//2:]]
    #training ml algos
    Cs = [10**i for i in range(-10,11)]
    Ks = list(range(1,11))
    #train C:
    print("svm & logreg C algos training...")
    for c in Cs:
        m_svm = svm_classifier(train_data,train_lbls, c)
        scoresSVM.append(m_svm.score(validation_data,validation_lbls))
        m_log_reg = log_reg_classifier(train_data,train_lbls,c)
        scoresLogReg.append(m_log_reg.score(validation_data,validation_lbls))
    #train K:
    print("KNN K algos training...")
    for k in Ks:
        m_knn = knn_classifier(train_data,train_lbls,k)
        scoresKNN.append(m_knn.score(validation_data,validation_lbls))
    save_plt_scores(Cs,"C",scoresSVM, "SVM scores","SVM scores as a function of C (on vaildation data)")
    save_plt_scores(Cs,"C",scoresLogReg, "Logistic Regression scores","Logistic Regression scores as a function of C (on vaildation data)")
    save_plt_scores(Ks,"K",scoresKNN, "KNN scores","KNN scores as a function of K (on vaildation data)", False)

def test_train_times(inputFolderCKData):
    timesSVM = []
    timesLogReg = []
    timesKNN = []
    print("Start testing...")
    (facial_landmarks_data, facial_landmarks_lbls) = dataset_from_ck(inputFolderCKData)
    facial_landmarks_lbls = np.array(facial_landmarks_lbls)
    features_df = extract_features_forall(facial_landmarks_data)
    Ds = [i for i in range(100, 1600, 100)]
    print("Measuring times...")
    for d in Ds:
        pca = dimension_reduction_pca(features_df, d)
        features_red = pca.transform(features_df)
        t1 = time.time()
        svm_classifier(features_red,facial_landmarks_lbls)
        t2 = time.time()
        log_reg_classifier(features_red,facial_landmarks_lbls)
        t3 = time.time()
        knn_classifier(features_red,facial_landmarks_lbls)
        t4 = time.time()
        timesSVM.append(t2-t1)
        timesLogReg.append(t3-t2)
        timesKNN.append(t4-t3)
    # timesSVM = np.array(timesSVM) / len(features_red)
    # timesLogReg = np.array(timesLogReg) / len(features_red)
    # timesKNN = np.array(timesKNN) / len(features_red)
    save_plt_scores(Ds,"d",timesSVM, "SVM train time","SVM train time as a function of d", False)
    save_plt_scores(Ds,"d",timesLogReg, "Logistic Regression train time","Logistic Regression train time as a function of d", False)
    save_plt_scores(Ds,"d",timesKNN, "KNN train time","KNN train time as a function of d", False)

def plot_3_principal_components(inputFolderCKData):
    plot_title = "3 Principal Components Scatter"
    colors_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple', 5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}
    (facial_landmarks_data, facial_landmarks_lbls) = dataset_from_ck(inputFolderCKData)
    facial_landmarks_lbls = np.array(facial_landmarks_lbls)
    features_df = extract_features_forall(facial_landmarks_data)
    pca = dimension_reduction_pca(features_df, 3)
    features_red = pca.transform(features_df)
    features_red = features_red.transpose()
    colors = np.array([colors_dict[x] for x in facial_landmarks_lbls])
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features_red[0], features_red[1], features_red[2], c=colors)
    fig.savefig(plot_title+'.png')


def test_train_NN_times(inputFolderCKData):
    print("Start testing...")
    (facial_landmarks_data, facial_landmarks_lbls) = dataset_from_ck(inputFolderCKData)
    facial_landmarks_lbls = np.array(facial_landmarks_lbls)
    features_df = extract_features_forall(facial_landmarks_data)
    features_df["expression"] = facial_landmarks_lbls
    for i in range(len(EMOTIONS)):
        features_df["is_{0:s}".format(EMOTIONS[i])] = (features_df["expression"] == i)
    features_df = features_df.drop("expression", axis=1)
    #features_df.to_csv("face.csv")
    return features_df


#######################################################################################
##############            RUN                                              ############
#######################################################################################
    
#test_images_flow(r"C:\Users\DELL1\Documents\studies\FinalProject\facial-landmarks\facial-landmarks\images")
#test_ml_algos_on_ck(r"C:\Users\DELL1\Documents\studies\FinalProject\Datatsets\CK+\sorted_set")
#plot_3_principal_components(r"C:\Santos\TAU\Final\Datasets\CK+\sorted_set - CK+")
sort_sample_affectnet(r"C:\Users\DELL1\Documents\studies\FinalProject\Datatsets\AffectNet\Manually_Annotated\FirstTrain", r"C:\Users\DELL1\Documents\studies\FinalProject\Datatsets\AffectNet\\Manually_Annotated\FirstTrain.csv")
