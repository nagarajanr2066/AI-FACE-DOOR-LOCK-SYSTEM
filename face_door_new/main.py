from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="face_door_open"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff2=open("img.txt","w")
    ff2.write("0")
    ff2.close()

    if request.method=='GET':
        act = request.args.get('act')
        

    return render_template('index.html',msg=msg,act=act)



@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        
        uname=request.form['uname']
        pass1=request.form['pass']

        
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM fd_register where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM fd_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO fd_register(id, name, mobile, email,uname,pass,rdate,rid,utype) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, uname, pass1, rdate,maxid,'admin')
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            return redirect(url_for('add_photo',vid=maxid))
            
            
        else:
            result="User already Exist!"
    return render_template('register.html',result=result)

@app.route('/login', methods=['POST','GET'])
def login():
    result=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        uname = request.form['uname']
        pass1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM fd_register where uname=%s && pass=%s",(uname,pass1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            #session['username'] = uname
            ff1=open("log.txt","w")
            ff1.write(uname)
            ff1.close()
            return redirect(url_for('admin')) 
        else:
            result="Your logged in fail!!!"
                
    
    return render_template('login.html',result=result)

@app.route('/admin',methods=['POST','GET'])
def admin():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id FROM fd_register where uname=%s",(uname,))
    rid = mycursor.fetchone()[0]

    mycursor.execute("SELECT * FROM fd_register where rid=%s",(rid,))
    result = mycursor.fetchall()

    return render_template('admin.html',result=result)

@app.route('/add_member',methods=['POST','GET'])
def add_member():
    msg=""
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()

    if 'username' in session:
        uname = session['username']
    #uname="vijay"
    print(uname)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id FROM fd_register where uname=%s",(uname,))
    rid = mycursor.fetchone()[0]
    
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        detail=request.form['detail']
        

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()
        
        
        mycursor.execute("SELECT max(id)+1 FROM fd_register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        str1=str(maxid)
        usr="u"+str1
        pass1='1234'
        utype='user'
        
        sql = "INSERT INTO fd_register(id, name, mobile, detail, uname, pass, rdate, rid, utype) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid, name, mobile, detail, usr, pass1, rdate, rid, utype)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        
        
        return redirect(url_for('add_photo2',vid=maxid)) 
        

    return render_template('add_member.html',msg=msg)

@app.route('/add_pin',methods=['POST','GET'])
def add_pin():
    msg=""
    act=""
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()

    if 'username' in session:
        uname = session['username']
    #uname="vijay"
    print(uname)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id FROM fd_register where uname=%s",(uname,))
    rid = mycursor.fetchone()[0]

    if request.method=='GET':
        act = request.args.get('act')
    if request.method=='POST':
        act = request.args.get('act')
        
        if act=="1":
            pin=request.form['pin']
            mycursor.execute('update fd_register set pin=%s WHERE uname = %s', (pin, uname))
            mydb.commit()
        elif act=="2":
            pin=randint(1000,9999)
            mycursor.execute('update fd_register set pin=%s WHERE uname = %s', (pin, uname))
            mydb.commit()

    mycursor.execute("SELECT * FROM fd_register where uname=%s",(uname,))
    data = mycursor.fetchone()

    return render_template('add_pin.html',msg=msg,data=data)

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    ff2=open("photo.txt","w")
    ff2.write("2")
    ff2.close()
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        cursor = mydb.cursor()

        cursor.execute('delete from fd_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM fd_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO fd_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update fd_register set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('login',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM fd_register")
    data = cursor.fetchall()
    return render_template('add_photo.html',data=data, vid=vid)

@app.route('/add_photo2',methods=['POST','GET'])
def add_photo2():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        cursor = mydb.cursor()

        cursor.execute('delete from fd_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM fd_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO fd_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update fd_register set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('admin',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM fd_register")
    data = cursor.fetchall()
    return render_template('add_photo2.html',data=data, vid=vid)

@app.route('/view_cus',methods=['POST','GET'])
def view_cus():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    value = mycursor.fetchall()
    return render_template('view_cus.html', result=value)

@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            ##Segmentation
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            
            ##Feature Extraction
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
            shutil.copy('static/images/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid)

@app.route('/view_photo1',methods=['POST','GET'])
def view_photo1():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo1.html', result=value,vid=vid)

@app.route('/view_photo2',methods=['POST','GET'])
def view_photo2():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo2.html', result=value,vid=vid)    

@app.route('/view_photo3',methods=['POST','GET'])
def view_photo3():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo3.html', result=value,vid=vid)

@app.route('/view_photo4',methods=['POST','GET'])
def view_photo4():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo4.html', result=value,vid=vid)

@app.route('/message',methods=['POST','GET'])
def message():
    vid=""
    name=""
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT name FROM register where id=%s",(vid, ))
        name = mycursor.fetchone()[0]
    return render_template('message.html',vid=vid,name=name)


###Segmentation using RNN
def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)
###Feature extraction & Classification
def DCNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted and classified')
        else:
                print('none')

@app.route('/userhome')
def userhome():
    uname=""
    if 'username' in session:
        uname = session['username']
        

    name=""
    
   
    

    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor1.fetchone()
    print(value)
    name=value[1]  
        
    return render_template('userhome.html',name=name)

'''@app.route('/deposit')
def deposit():
    return render_template('deposit.html')
@app.route('/deposit_amount',methods=['POST','GET'])
def deposit_amount():
    if request.method=='POST':
        name=request.form['name']
        accountno=request.form['accno']
        amount=request.form['amount']
        today = date.today()
        rdate = today.strftime("%b-%d-%Y")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM event")
        maxid = mycursor.fetchone()[0]
        sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, name, accountno, amount, rdate)
        mycursor.execute(sql, val)
        mydb.commit()   
    return render_template('userhome.html')'''

'''@app.route('/withdraw')
def withdraw():

    
    return render_template('withdraw.html')'''

@app.route('/verify_face',methods=['POST','GET'])
def verify_face():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    if 'username' in session:
        uname = session['username']
    print(uname)
    shutil.copy('faces/f1.jpg', 'static/f1.jpg')
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    name=account[1]
    mobile=account[3]
    print(mobile)
    email=account[4]
    vid=account[0]
    #shutil.copy('faces/f1.jpg', 'http://iotcloud.co.in/testsms/upload/f1.jpg')
    if act=="1":
        shutil.copy('faces/f1.jpg', 'faces/s1.jpg')
        cutoff=10
        img="v"+str(vid)+".jpg"
        cursor.execute('SELECT * FROM vt_face WHERE vid = %s', (vid, ))
        dt = cursor.fetchall()
        for rr in dt:
            hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
            hash1 = imagehash.average_hash(Image.open("faces/s1.jpg"))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=4:
                ss="ok"
                break
            else:
                ss="no"
        if ss=="ok":
            act="2"
            print("correct person")
            #return redirect(url_for('login', msg=msg))
        else:
            act="1"
            print("wrong person")
            #xn=randint(1000, 9999)
            #otp=str(xn)
            
            #cursor1 = mydb.cursor()
            #cursor1.execute('update register set otp=%s WHERE card = %s', (otp, uname))
            #mydb.commit()

            '''mess="Someone Access your account"
            url2="http://localhost/atm/img.txt"
            ur = urlopen(url2)#open url
            data1 = ur.read().decode('utf-8')

           
            idd=int(data1)
            url="http://iotcloud.co.in/testsms/sms.php?sms=link&name="+name+"&mess="+mess+"&mobile="+str(mobile)+"&id="+str(idd)
            webbrowser.open_new(url)'''
            
                
    return render_template('verify_face.html',msg=msg,act=act)


@app.route('/cap',methods=['POST','GET'])
def cap():
    msg=""
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    
    gf=open("file.txt","r")
    file=gf.read()
    gf.close()

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM fd_register where uname=%s',(uname,))
    rss = cursor.fetchone()
    mobile=rss[2]
    pin=rss[13]
    
    return render_template('cap.html',msg=msg,fn=file,pin=pin,bc=bc)

@app.route('/verify',methods=['POST','GET'])
def verify():
    msg=""
    #data1="4"
    url2="http://localhost/door/log.txt"
    ur = urlopen(url2)#open url
    data1 = ur.read().decode('utf-8')
    return render_template('verify.html',msg=msg,data=data1)

@app.route('/page',methods=['POST','GET'])
def page():
    
    act=""
    msg=""
    fn=""

    cursor = mydb.cursor(buffered=True)
    ut='admin'
    cursor.execute("SELECT id FROM fd_register where utype=%s order by id desc",(ut,))
    rid = cursor.fetchone()[0]

    

    ff1=open("photo.txt","w")
    ff1.write("3")
    ff1.close()
        
    
    ff31=open("img.txt","r")
    getface=ff31.read()
    ff31.close()
    print("face="+getface)
    m=int(getface)

    cursor.execute("SELECT max(id)+1 FROM fd_register")
    maxid2 = cursor.fetchone()[0]
    if maxid2 is None:
        maxid2=1

    ff=open("user.txt","w")
    ff.write(str(maxid2))
    ff.close()
    
    if m>1:
        act="1"
        print("Face Detected")
        gf=open("file.txt","r")
        file=gf.read()
        gf.close()

        cursor.execute('delete from fd_temp')
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM fd_temp")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=str(maxid2)+"_"+str(i)+".jpg"
            sql = "INSERT INTO fd_temp(id, vface) VALUES (%s, %s)"
            val = (maxid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
        if file=="":
            a=1
        else:
            a=2
            #os.remove("static/photo/"+file)
        
        nn=randint(111,999)
        fn=str(nn)+".jpg"
        gf1=open("file.txt","w")
        gf1.write(fn)
        gf1.close()

        gf2=open("det.txt","w")
        gf2.write('0')
        gf2.close()
        
        shutil.copy('faces/f1.jpg', 'static/photo/'+fn)
        return redirect(url_for('analyze', act=act, fn=fn))
        
    else:
        print("No Face")

    return render_template('page.html',act=act, fn=fn)
#########################

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    msg=""
    name=""
    mobile=""
    pin=""
    ss=""
    dst=""
    data1=""
    act=request.args.get('act')
    fn=request.args.get('fn')
    
    ff31=open("img.txt","r")
    getface=ff31.read()
    ff31.close()

    gf=open("file.txt","r")
    file=gf.read()
    gf.close()

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()

    
    cutoff=10
    
    cursor = mydb.cursor(buffered=True)
    cursor.execute('SELECT * FROM fd_face')
    dt = cursor.fetchall()

    ut='admin'
    cursor.execute("SELECT id FROM fd_register where utype=%s order by id desc",(ut,))
    rid = cursor.fetchone()[0]

    cursor.execute('SELECT * FROM fd_register where id=%s',(rid,))
    rss = cursor.fetchone()
    mobile=rss[2]
    pin=rss[13]
        
    for rr in dt:
        hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
        hash1 = imagehash.average_hash(Image.open("static/photo/"+file))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<cutoff:
            vid=rr[1]
            print("vid="+str(vid))
            
            ss="ok"
            break
        else:
            ss="no"

    print(ss)
    if ss=="ok":
        cursor.execute('SELECT * FROM fd_register where id=%s',(vid,))
        rw = cursor.fetchone()
        name=rw[1]
        
        dst="1"
        msg="Welcome "+name
        
    else:
        dst="2"
        vid="0"
        name="Unknown"
        msg="Unknown Person"


    if act=="1":
        cursor.execute("SELECT max(id)+1 FROM fd_history")
        maxid = cursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO fd_history(id, rid, vid, name, vface) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, rid, vid, name, fn)
        cursor.execute(sql, val)
        mydb.commit()

        if dst=="2":
            a=1
            mess="Door Access"
            url2="http://localhost/door/img.txt"
            ur = urlopen(url2)#open url
            data1 = ur.read().decode('utf-8')

            print("msg mobile "+str(mobile))
            idd=int(data1)
            url="http://iotcloud.co.in/testsms/sms.php?sms=linkdoor1&name=User&mess="+mess+"&mobile="+str(mobile)+"&id="+str(idd)+"&bc="+bc
            webbrowser.open_new(url)


    if dst=="1":
        print("yes")
        
    else:
        
        url2="http://localhost/door/log.txt"
        ur = urlopen(url2)#open url
        data1 = ur.read().decode('utf-8')
        dt=data1.split('-')
        print("data1="+data1)
        if dt[0]=="2":
            dst="3"
        if dt[0]=="3":
            dst="4"
            nam=dt[2]
            mob=dt[3]
            deta=dt[4]
            now = datetime.datetime.now()
            rdate=now.strftime("%d-%m-%Y")
            mycursor = mydb.cursor()
            
            
            mycursor.execute("SELECT max(id)+1 FROM fd_register")
            maxid3 = mycursor.fetchone()[0]
            if maxid3 is None:
                maxid3=1

            str1=str(maxid3)
            usr="u"+str1
            pass1='1234'
            utype='user'
            
            sql = "INSERT INTO fd_register(id, name, mobile, detail, uname, pass, rdate, rid, utype) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid3, nam, mob, deta, usr, pass1, rdate, rid, utype)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
            ######
            mycursor.execute('SELECT * FROM fd_temp')
            rs1 = mycursor.fetchall()
            for rs2 in rs1:
                fc2=rs2[1]
                shutil.copy('static/temp/'+fc2, 'static/frame/'+fc2)
                mycursor.execute("SELECT max(id)+1 FROM fd_face")
                maxid4 = mycursor.fetchone()[0]
                if maxid4 is None:
                    maxid4=1
                
                sql = "INSERT INTO fd_face(id, vid, vface) VALUES (%s, %s, %s)"
                val = (maxid4, maxid3, fc2)
                print(sql)
                mycursor.execute(sql, val)
                mydb.commit()
                
            
            print("save")
        if dt[0]=="4":
            dst="3"
            print("close")
    
    
        
    if act=="4":
        ff2=open("img.txt","w")
        ff2.write("0")
        ff2.close()
        
    return render_template('analyze.html',act=act,fn=fn,msg=msg,dst=dst,data1=data1)

@app.route('/history',methods=['POST','GET'])
def history():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id FROM fd_register where uname=%s",(uname,))
    rid = mycursor.fetchone()[0]
    

    
    mycursor.execute('SELECT * FROM fd_history where rid=%s order by id desc',(rid, ))
    data = mycursor.fetchall()

    return render_template('history.html',data=data)
    
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))

def gen(camera):
    
    while True:
        frame = camera.get_frame()

        
        ########
        '''try:
            cutoff=20

            f1=open("img.txt","r")
            imgcnt=f1.read()
            f1.close()
            vid=""
            uarr=[]
            if imgcnt is None:
                imgcnt=""
            else:
                mcnt=int(imgcnt)-1
                if mcnt>0:
                    k=0
                    
                    while k<mcnt:
                        k+=1
                        fn="f"+str(k)+".jpg"
                        #print(fn)
                        path="faces/"+fn
                        path1='static/photo/'+fn
                        shutil.copy('faces/'+fn, 'static/photo/'+fn)
                        
                        cursor.execute('SELECT * FROM fd_face')
                        dt = cursor.fetchall()
                        for rr in dt:
                            hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
                            hash1 = imagehash.average_hash(Image.open(path1))
                            cc1=hash0 - hash1
                            print("cc="+str(cc1))
                            if cc1<cutoff:
                                ss="ok"
                                vid=str(rr[1])
                                cursor.execute('SELECT * FROM fd_face where vid=%s limit 0,1',(vid, ))
                                vdt = cursor.fetchone()
                                vf=vdt[2]
                                #print("vid="+vid)
                                cursor.execute('update fd_register set detect=1,vface=%s WHERE id = %s', (vf, vid))
                                mydb.commit()
                                
                                
                            else:
                                ss="no"
                        uarr.append(vid)
                    txt=','.join(uarr)      
                    ff2=open("u1.txt","w")
                    ff2.write(txt)
                    ff2.close()
        except:
            print("try1")'''
        #################
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
