################
# Installation #
################

This project is based on a conda virtual 
environment with Python==3.7.6. Old versions
of TensorFlow==1.15.0 and TFLearn==0.3.2 are
needed and other dependences are up-to-date.

Step 1: create a conda virtual environment

conda create env -n chatbot python==3.7
[AFTER INSTALLATION]
conda activate chatbot

Step 2: install dependent packages

conda install numpy
conda install flask
conda install flask-socketio
pip install tensorflow==1.15.0
pip install tflearn==0.3.2

Step 3: launch the service

python main.py

Step 4: chat with the bot

http://localhost:5000/

#################
# Documentation #
#################

This project is a chatbot with a web user
interface. Chatbot is build on a simple
bags-of-words model which has pretty good
performance when facing specific kinds of
questions rather than free chat. A model is
trained to be a multilayer percepton to tell
the most-likely class of a given question and
find the corresponding answers in input maps.
After model is trained, a web socket service
is opened at localhost:5000. The interaction
is realized by flask and JavaScript. For ui
design, check "./application/" for more info.
The JavaScript is in "./application/static/"
and the design patterns files (.html files)
are in "./application/templates". After the
user enters the message, it will be recorded
to a SQLite file "./messages.db" for analysis.

############
# Training #
############

Inside is a pre-trained model. Check the
style of training set in "intents.json".
The pre-trained model is save in files
beginning with "model.tflearn".



Appendix: full list of packages:

Package              Version
-------------------- -------------------
absl-py              0.8.1
astor                0.8.0
certifi              2019.11.28
Click                7.0
Flask                1.1.1
Flask-SocketIO       4.2.1
gast                 0.2.2
google-pasta         0.1.8
grpcio               1.16.1
h5py                 2.10.0
itsdangerous         1.1.0
Jinja2               2.11.1
Keras                2.2.4
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
Markdown             3.1.1
MarkupSafe           1.1.1
mkl-fft              1.0.15
mkl-random           1.1.0
mkl-service          2.3.0
nltk                 3.4.5
numpy                1.18.1
opt-einsum           3.1.0
Pillow               7.0.0
pip                  20.0.2
protobuf             3.11.2
pyreadline           2.1
python-dotenv        0.11.0
python-engineio      3.11.2
python-socketio      4.4.0
PyYAML               5.3
scipy                1.4.1
setuptools           45.1.0.post20200127
six                  1.14.0
tensorboard          2.0.0
tensorflow           1.15.0
tensorflow-estimator 1.15.1
termcolor            1.1.0
tflearn              0.3.2
Werkzeug             0.16.1
wheel                0.34.2
wincertstore         0.2
wrapt                1.11.2

