import streamlit as st
import streamlit_authenticator as stauth
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.let_it_rain import rain
from streamlit_option_menu import option_menu
from streamlit_card import card
import database as db
import requests
import webbrowser
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import cv2
import pytesseract
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')
import re
import warnings

users = db.fetch_all_users()

usernames = [user['key'] for user in users]
names = [user['name'] for user in users]
hashed_passwords = [user['password'] for user in users]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'vetassist', 'abcdef', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login to MedBot', 'main')

if authentication_status == False:
    st.error('Username/Password is incorrect!')

if authentication_status == None:
    st.warning('Please enter your username and password to login!')

if authentication_status:
    
    with st.sidebar:
        st.sidebar.title(f'Welcome, {name}')
        selected = option_menu(
            menu_title = 'Main Menu',
            options = ['Home', 'Profile', 'Recommendation'],
            icons = ['house', 'person', 'check-all'],
            menu_icon = 'cast',
            default_index = 0
        )
        authenticator.logout('Logout', 'sidebar')
    
    if selected == 'Home':

        st.header('Predicting Hospital Readmissions')

        add_vertical_space(3)

        st.image('https://www.eagletelemedicine.com/wp-content/uploads/2020/11/5-Ways-LI.jpg', width=700)

        add_vertical_space(3)

        st.subheader('This tools aims to assist you in planning your time and finances in cases you may need to be readmitted to the hospital. This is accomplished by leveraging Machine Learning principles to predict emergency readmissions within 30 days. All we require is a photo of your discharge summary. Login, upload and get results!')

        add_vertical_space(1)

        st.write('Visit the following links for more information:')

        url1 = "https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/readmissions-reduction-program"
        url2 = 'https://www.ahrq.gov/topics/hospital-readmissions.html'
        url3 = 'https://www.uptodate.com/contents/hospital-discharge-and-readmission'
        url4 = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4731880/'
        url5 = 'https://www.commonwealthfund.org/publications/newsletter-article/focus-preventing-unnecessary-hospital-readmissions'

        st.write("[Hospital Readmissions Reduction Program (HRRP)](%s)" % url1)
        st.write("[Hospital Readmissions](%s)" % url2)
        st.write("[Hospital discharge and readmission](%s)" % url3)
        st.write("[Why do patients keep coming back? ](%s)" % url4)
        st.write("[In Focus: Preventing Unnecessary Hospital Readmissions](%s)" % url5)

        if 'eligibility_vettec' not in st.session_state:
                st.session_state.eligibility_vettec = False
        
        if 'eligibility_gibill' not in st.session_state:
                st.session_state.eligibility_gibill = False
    
    if selected == 'Profile':

        st.header("Profile")

        st.subheader("Patient Identification Information")

        col7, col8 = st.columns([1,1])

        with col7:

            st.markdown('Patient Information:')

            if 'name' not in st.session_state:
                st.session_state.name = ''
            st.session_state.name = st.text_input(
                "Name:",
                value = st.session_state.name
            )

            if 'email' not in st.session_state:
                st.session_state.email = ''
            st.session_state.email = st.text_input(
                "Email:",
                value = st.session_state.email
            )

            if 'phone' not in st.session_state:
                st.session_state.phone = ''
            st.session_state.phone = st.text_input(
                "Phone Number:",
                value = st.session_state.phone
            )

            if 'address' not in st.session_state:
                st.session_state.address = ''
            st.session_state.address = st.text_input(
                "Address:",
                value = st.session_state.address
            )

            if 'gender' not in st.session_state:
                st.session_state.gender = ''
            st.session_state.gender = st.radio(
                "Gender:",
                options=["Male", "Female", 'Prefer not to share'],
            )

            if 'dob' not in st.session_state:
                st.session_state.dob = ''
            st.session_state.dob = st.date_input("Date of Birth:")

        with col8:

            st.markdown('Emergency Contact Information:')

            if 'ename' not in st.session_state:
                st.session_state.ename = ''
            st.session_state.ename = st.text_input(
                "Name:",
                value = st.session_state.ename,
            )

            if 'ephone' not in st.session_state:
                st.session_state.ephone = ''
            st.session_state.ephone = st.text_input(
                "Phone Number:",
                value = st.session_state.ephone, key=45
            )

            if 'eaddress' not in st.session_state:
                st.session_state.eaddress = ''
            st.session_state.eaddress = st.text_input(
                "Address:",
                value = st.session_state.eaddress, key=56
            )

        add_vertical_space(5)

        st.subheader("Medication Information")

        if 'medication' not in st.session_state:
            st.session_state.medication = ''
        st.session_state.medication = st.radio(
            "Are you on any current medication?",
            options=["Yes", "No"],
        )

        if 'discharge_med' not in st.session_state:
            st.session_state.discharge_med = ''
        st.session_state.discharge_med = st.radio(
            "Were you on medication during your discharge period?",
            options=["Yes", "No"],
        )

        def admission_info_func():
            st.session_state.date_admitted = st.date_input("Date Admitted:", key=1)
            st.session_state.date_discharged = st.date_input("Date Discharged:", key=2)
            st.session_state.doctor_comments = st.text_input("Doctor Comments:")
            st.session_state.patient_feedback = st.text_input("Patient Feedback:")
            st.session_state.voluntary_adm = st.radio(
                "Were you involuntarily admitted as an emergency case?",
                options=["Yes", "No"],
            )

        work_expander = st.expander("Admission Information:", expanded=True)
        with work_expander:
            st.button("(+) Add admission information", on_click=admission_info_func())


        col11, col12, col13 = st.columns([1.2,1,1])

        with col11:
            st.write(' ')

        with col12:
            st.write("[Reach out to Hospital](%s)" % 'https://www.hopkinsmedicine.org/the_johns_hopkins_hospital/')

        with col13:
            st.write(' ')
        
        add_vertical_space(2)

        col9, col10, col14 = st.columns([4,3,1])
        with col9:
            st.button('Save')

        with col10:
            st.write(' ')

        with col14:
            if st.button('Cancel'):
                st.session_state.name = ''
                st.session_state.email = ''
                st.session_state.phone = ''
                st.session_state.address = ''
                st.session_state.loc_ref_pr = ''
                st.session_state.reloc_pr = ''
                st.session_state.summary = ''
                st.session_state.skills = ''
                st.session_state.work_ex = ''
                st.session_state.mil_hon = ''
                st.session_state.education = ''
                st.session_state.references = ''
                switch_page('Home')



    
    if selected == 'Recommendation':

        st.header('Readmission Recommendation Tab')
        
        st.subheader('Upload Patient Discharge Summary')

        file_type = st.radio(
            "What is the file type of your report: ",
            ('jpg', 'png'))
        
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

        if file_type == 'jpg':
            st.write('You selected your report is a jpg file.')

            uploaded_file = st.file_uploader('Choose your patient discharge report', type="jpg")
                
            if st.button('Get Prediction'):
                
                img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
                doc = pytesseract.image_to_string(img)
                    
                processed_report = doc.replace('\n',' ')
                processed_report = processed_report.replace('\r',' ')

                stopwords=nltk.corpus.stopwords.words('english')

                def treat_text(text):
                    edited_text=re.sub('\W'," ",text) #replace any sumbol with whitespace
                    edited_text=re.sub("  "," ",edited_text) #replace double whitespace with single whitespace
                    edited_text=edited_text.split(" ") #split the sentence into array of strings
                    edited_text=" ".join([char for char in edited_text if char!= ""]) #remove any empty string from text
                    edited_text=edited_text.lower() #lowercase
                    edited_text=re.sub('\d+',"",edited_text) #Removing numerics
                    edited_text = re.sub('_', '', edited_text)
                    edited_text=re.split('\W+',edited_text) #spliting based on whitespace or whitespaces
                    edited_text=" ".join([word for word in edited_text if word not in stopwords])
                    return edited_text
                
                clean = treat_text(processed_report)

                report_list = []
                report_list.append(clean)

                def tokenizer(text):
        
                    punc_list = string.punctuation+'0123456789'
                    t = str.maketrans(dict.fromkeys(punc_list, " "))
                    text = text.lower().translate(t)
                    tokens = word_tokenize(text)
                    return tokens

                with open("count_vect.pkl", 'rb') as file1:  
                    loaded_vect = pickle.load(file1)
                
                with open("log_reg_model.pkl", 'rb') as file2:  
                    loaded_model = pickle.load(file2)

                X_report = loaded_vect.transform(report_list)

                y_report = loaded_model.predict_proba(X_report)[:,1]

                score = float(str(y_report[0]*100))

                st.success('The patient has a ' + "%.2f" %score + ' percent chances of readmission within 30 days.')
                st.warning('We recommend taking action if the above percentage is greater than 50%.')
                add_vertical_space(3)
                st.warning('Please upload a patient discharge report if not uploaded.')
                add_vertical_space(3)
        
        if file_type == 'png':
            st.write('You selected your report is a png file.')

            uploaded_file = st.file_uploader('Choose your patient discharge report', type="png")
                
            if st.button('Get Prediction'):
                
                img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
                doc = pytesseract.image_to_string(img)
                    
                processed_report = doc.replace('\n',' ')
                processed_report = processed_report.replace('\r',' ')

                stopwords=nltk.corpus.stopwords.words('english')

                def treat_text(text):
                    edited_text=re.sub('\W'," ",text) #replace any sumbol with whitespace
                    edited_text=re.sub("  "," ",edited_text) #replace double whitespace with single whitespace
                    edited_text=edited_text.split(" ") #split the sentence into array of strings
                    edited_text=" ".join([char for char in edited_text if char!= ""]) #remove any empty string from text
                    edited_text=edited_text.lower() #lowercase
                    edited_text=re.sub('\d+',"",edited_text) #Removing numerics
                    edited_text = re.sub('_', '', edited_text)
                    edited_text=re.split('\W+',edited_text) #spliting based on whitespace or whitespaces
                    edited_text=" ".join([word for word in edited_text if word not in stopwords])
                    return edited_text
                
                clean = treat_text(processed_report)

                report_list = []
                report_list.append(clean)

                def tokenizer(text):
        
                    punc_list = string.punctuation+'0123456789'
                    t = str.maketrans(dict.fromkeys(punc_list, " "))
                    text = text.lower().translate(t)
                    tokens = word_tokenize(text)
                    return tokens

                with open("count_vect.pkl", 'rb') as file1:  
                    loaded_vect = pickle.load(file1)
                
                with open("log_reg_model.pkl", 'rb') as file2:  
                    loaded_model = pickle.load(file2)

                X_report = loaded_vect.transform(report_list)

                y_report = loaded_model.predict_proba(X_report)[:,1]

                score = float(str(y_report[0]*100))

                st.success('The patient has a ' + "%.2f" %score + ' percent chances of readmission within 30 days.')
                st.warning('We recommend taking action if the above percentage is greater than 50%.')
                add_vertical_space(3)
                st.warning('Please upload a patient discharge report if not uploaded.')
                add_vertical_space(3)