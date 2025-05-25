from flask import Flask, request, render_template, redirect, url_for
import sqlite3
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/UserHome')
def UserHome():
    return render_template('user/userhome.html')

# Route to add a new record (INSERT) student data to the database
@app.route("/addrec", methods = ['POST', 'GET'])
def addrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']
            # Connect to SQLite3 database and execute the INSERT
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("INSERT INTO students (name, loginid, email,  password, addr, city, zip) VALUES (?,?,?,?,?,?,?)",(nm, loginid, email, password, addr, city, zip))
                con.commit()
                msg = "Record successfully added to database"
        except:
            con.rollback()
            msg = "Error in the INSERT"
        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)       
        
        
@app.route("/enternew")
def enternew():
    return render_template("student.html")


@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM students")

    rows = cur.fetchall()
    con.close()
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)



@app.route("/edit", methods=['POST','GET'])
def edit():
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            id = request.form['id']
            # Connect to the database and SELECT a specific rowid
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            cur.execute("SELECT rowid, * FROM students WHERE rowid = " + id)

            rows = cur.fetchall()
        except:
            id=None
        finally:
            con.close()
            # Send the specific record of data to edit.html
            return render_template("edit.html",rows=rows)
        
        

# # Route used to execute the UPDATE statement on a specific record in the database
# @app.route("/editrec", methods=['POST','GET'])
# def editrec():
#     # Data will be available from POST submitted by the form
#     if request.method == 'POST':
#         try:
#             # Use the hidden input value of id from the form to get the rowid
#             rowid = request.form['rowid']
#             nm = request.form['nm']
#             addr = request.form['add']
#             city = request.form['city']
#             zip = request.form['zip']
#             loginid = request.form['loginid']
#             email = request.form['email']
#             password = request.form['password']
#             # UPDATE a specific record in the database based on the rowid
#             with sqlite3.connect('database.db') as con:
#                 cur = con.cursor()
#                 cur.execute("UPDATE students SET name='"+nm+"', addr='"+addr+"', city='"+city+"', zip='"+zip+"', loginid='"+loginid+"',email='"+email+"', password='"+password+"' WHERE rowid="+rowid)

#                 con.commit()
#                 msg = "Record successfully edited in the database"
#         except:
#             con.rollback()
#             msg = "Error in the Edit: UPDATE students SET name="+nm+", addr="+addr+", city="+city+", zip="+zip+" loginid="+loginid+",email="+email+", password="+password+" WHERE rowid="+rowid

#         finally:
#             con.close()
#             # Send the transaction message to result.html
#             return render_template('result.html',msg=msg)
        
        

# Route used to execute the UPDATE statement on a specific record in the database
@app.route("/editrec", methods=['POST'])
def editrec():
    if request.method == 'POST':
        con = None
        try:
            print(request.form)
            rowid = request.form['rowid']
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']

            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("""
                    UPDATE students SET name=?, addr=?, city=?, zip=?, loginid=?, email=?, password=?
                    WHERE rowid=?
                """, (nm, addr, city, zip, loginid, email, password, rowid))

                con.commit()
                msg = "Record successfully edited in the database"
        except Exception as e:
            if con:
                con.rollback()
            msg = f"Error in the Edit: {str(e)}"
        finally:
            if con:
                con.close()
            return render_template('result.html', msg=msg)







# Route used to DELETE a specific record in the database    
@app.route("/delete", methods=['POST','GET'])
def delete():
    if request.method == 'POST':
        try:
             # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['id']
            # Connect to the database and DELETE a specific record based on rowid
            with sqlite3.connect('database.db') as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM students WHERE rowid="+rowid)

                    con.commit()
                    msg = "Record successfully deleted from the database"
        except:
            con.rollback()
            msg = "Error in the DELETE"

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)
        
        
        
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        loginid = request.form['loginid']
        password = request.form['password']
        print(loginid, password)
        query = "SELECT loginid, password FROM students WHERE loginid=? AND password=?"
        cursor.execute(query, (loginid, password))
        results = cursor.fetchall()        
        if not results:
            error_message = 'Invalid login credentials. Please try again.'
            return render_template('login.html', error=error_message)
        else:
            return render_template('user/userhome.html')
    return render_template('login.html')



import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
# Load the pre-trained model from the relative path in the media folder
model_path = os.path.join('media', 'my_model123.h5')
data = os.path.join('media','dataset')
model = tf.keras.models.load_model(model_path, compile=False)


    

@app.route('/training')
def training():
    # Adjust the path to the dataset directory inside the media folder
    dataset_dir = os.path.join(os.getcwd(), 'media', 'dataset')   
    
    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        return "Dataset directory does not exist. Please check the path."

    # Print directories and files once
    for root, dirs, files in os.walk(dataset_dir):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            print(os.path.join(root, name))
        break  
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(227, 227),
        batch_size=32,
        class_mode='binary',
        shuffle=True  
    )
    # Build the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(
        optimizer=RMSprop(lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=1,
        verbose=1
    )
    
    loss = history.history['loss'][-1]
    accuracy = history.history['accuracy'][-1]
    
    print(f'Final loss: {loss}')
    print(f'Final accuracy: {accuracy}')
    return render_template('user/training.html', loss=loss, accuracy=accuracy)
    


# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=filename))
    return render_template('user/index1.html')


@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = image.load_img(filepath, target_size=(227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale pixel values to [0, 1]

    prediction = model.predict(img_array)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0
    class_label = "Cracked" if predicted_class == 1 else "Uncracked"
    return render_template('user/result1.html', filename=filename, label=class_label)

    


if __name__ == '__main__':
    app.run(debug=True,port=8000)
