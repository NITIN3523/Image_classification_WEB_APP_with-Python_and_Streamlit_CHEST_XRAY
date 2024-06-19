import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify,set_background


## set background 
set_background('01.jpg')

## Set Title ##
st.title('Pneumonia Classification')

## set haeder ##
st.header('Please upload a CHest X-Ray')


## upload file ##
file = st.file_uploader('',type=['jpg','jpeg','png'])


## load classifier ##
model = load_model('Model/pneumonia_classifier.h5')


## load class image ##
with open('Model/labels.txt') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()] 
    f.close()

print(class_names)


## display image ##
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)


    ## classify image ##
    predction,score = classify(image,model,class_names)


    ## write clasification ##
    st.write('## {}'.format(predction))
    st.write('### Score :  {}'.format(score))
