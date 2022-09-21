from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from matplotlib.figure import Figure

app = Flask(__name__, template_folder='templates')

model = tf.keras.models.load_model('models\model_simple.h5')

@app.route('/')
def upload_file():
   return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
   if request.method == 'POST':
      img = Image.open(request.files['file'].stream).convert("L")
      img = img.resize((28,28))
      im2arr = np.array(img)
      im2arr = im2arr.reshape(1,28,28,1)
      y_pred = model.predict(im2arr)

      # Generate the figure **without using pyplot**.
      fig = Figure()
      ax = fig.subplots()
      ax.set_xticks(np.arange(10))
      ax.bar(np.arange(0,10), y_pred[0])
      buf = BytesIO()
      fig.savefig(buf, format="png")
      # Embed the result in the html output.
      data = base64.b64encode(buf.getbuffer()).decode("ascii")
      return f"Predicted Number:  {np.argmax(y_pred[0])} \n  Prediction probability:  {y_pred[0]}<img src='data:image/png;base64,{data}'/>"

@app.route('/test')
def test():
   return 'test'



if __name__ == '__main__':
   print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
   app.run(debug = True)