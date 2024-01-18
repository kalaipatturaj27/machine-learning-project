import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nlpaug.augmenter.word as aug
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance,ImageOps
import cv2
#from wordcloud import WordCloud
import pytesseract
from spellchecker import SpellChecker
#*************************************************************************************************************************************
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,recall_score,precision_score,auc,accuracy_score
import plotly.express as px

### models 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
import pickle
### customer recommendation
from itertools import chain
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

#*****************************************************************************************************************************************

st.set_page_config(page_title='Machine Learning Model',  layout="wide")
st.markdown(f'<h1 style="text-align: center;">Machine learning Model</h1>', unsafe_allow_html=True)

with st.sidebar:
    lis = ['select','prediction','nlp','image','customer recommendation']
    options =st.selectbox('',lis)
    
#***************************************************************************************************************************************************
def token(sample):
    word_tk=word_tokenize(sample)
    return word_tk
def stop_word(sample):
    word_tk=word_tokenize(sample)
    stop_words = stopwords.words('english')
    sr_list = ' '.join([i for i in word_tk if i not in stop_words ])
    return sr_list
def stem(sample):
    nlp = spacy.load("en_core_web_sm")
    text = stop_word(sample)
    doc = nlp(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token.text) for token in doc]
    
    return stemmed_tokens
    
def lemm(sample):
    nlp = spacy.load("en_core_web_sm")
    text = stop_word(sample)
    doc = nlp(text)
    lemmatized_tokens = ' '.join([token.lemma_ for token in doc])
    return lemmatized_tokens


def keyword(sample):
    vect = CountVectorizer(stop_words="english")
    ext_val = vect.fit_transform([sample])
    result = vect.get_feature_names_out()
    return result
    
def sentiment(sample):
    sent_ment = SentimentIntensityAnalyzer()
    sent_score = sent_ment.polarity_scores(sample)
    if sent_score['neg'] > 0.5:
        return 'Negative Sentence'
    elif sent_score['pos'] > 0.5:
        return 'Positive Sentence'
    elif sent_score['neu'] > 0.5:
        return 'Neutral Sentence'
    else:
        return sent_score
    
def aug_synon(sample):
    text_aug = aug.SynonymAug()
    text_list = []
    for i in range(0,2):
        au_text = text_aug.augment(sample)
        text_list.append(au_text)
    return text_list
def aug_ant(sample):
    text_aug = aug.AntonymAug()
    text_list = []
    for i in range(0,2):
        au_text = text_aug.augment(sample)
        text_list.append(au_text)
    return text_list
def both(sample):
    data= st.write(pd.DataFrame({
    'synonyms':aug_synon(sample),
    'antonyms':aug_ant(sample)}))
    return data  

#def word_cloud(sample):
   # width = st.sidebar.slider("**select width**", min_value=100, max_value=1000)
    #height = st.sidebar.slider("**select width**", min_value=100, max_value=1000)
    #color = st.selectbox('',['red','green','blue'])
    #w_c = WordCloud(width=width,height=height,background_color=color).generate(sample)
    
    #plt.figure(figsize=(14,6))
    #plt.imshow(w_c)
    #plt.axis('off')
    #st.pyplot(plt.gcf())
    
def pos(sample):
    tag_mapping = {
    'CC': 'Coordinating Conjunction',
    'CD': 'Cardinal Digit',
    'DT': 'Determiner',
    'EX': 'Existential There',
    'FW': 'Foreign Word',
    'IN': 'Preposition or Subordinating Conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, Comparative',
    'JJS': 'Adjective, Superlative',
    'LS': 'List Item Marker',
    'MD': 'Modal',
    'NN': 'Noun, Singular or Mass',
    'NNS': 'Noun, Plural',
    'NNP': 'Proper Noun, Singular',
    'NNPS': 'Proper Noun, Plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive Ending',
    'PRP': 'Personal Pronoun',
    'PRP$': 'Possessive Pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, Comparative',
    'RBS': 'Adverb, Superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, Base Form',
    'VBD': 'Verb, Past Tense',
    'VBG': 'Verb, Gerund or Present Participle',
    'VBN': 'Verb, Past Participle',
    'VBP': 'Verb, Non-3rd Person Singular Present',
    'VBZ': 'Verb, 3rd Person Singular Present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive Wh-pronoun',
    'WRB': 'Wh-adverb'
    }
    tokens =nltk.word_tokenize(sample)
    result=nltk.pos_tag(tokens)
    noun=[]
    non_noun=[]
    for i,j in result:
        if j in tag_mapping:
            
            st.write(i,':',tag_mapping[j])
            if j not in ['PRP','NN']:
                non_noun.append(i)
        
            else:
                noun.append(i)
    st.write('noun:',' ,'.join(noun))        
    st.write('non_noun:',' ,'.join(non_noun))
    return result
def spell(sample):
    checker = SpellChecker()
    sampletoken =word_tokenize(sample)
    result = ','.join([checker.correction(i) for i in sampletoken])
    return result
#************************************************************************************************************************************************
def save_card(uploaded):
    if uploaded is not None:
        uploaded_dir = os.path.join(os.getcwd(), "uploaded_images")
        os.makedirs(uploaded_dir, exist_ok=True)
    
        with open(os.path.join(uploaded_dir, uploaded.name), "wb") as f:
            f.write(uploaded.getbuffer())
    
        saved_image_path = os.path.join(uploaded_dir, uploaded.name)
        
        
    
        image = Image.open(saved_image_path)
    
        plt.rcParams['figure.figsize'] = (5, 5)
        plt.axis('off')
        plt.imshow(image)
        st.pyplot(plt.gcf())
        return saved_image_path
    else:
        st.warning("Please upload a file.")
        return None
#******************************************************
def propertise(image):
    #saved_img = os.getcwd() + "\\" + "uploaded_images" + "\\" + uploaded.name
    #image = cv2.imread(image)
    image = Image.open(image)
    st.title('Propertise of Image')   
    
    image_array = np.array(image)
    properties_string = f"Image Mode: {image.mode}\n\nImage Format: {image.format}\n\nImage Size: {image.size}\n\nImage Shape: {image_array.shape}"
    return properties_string
#**************************************************************    
def convert(image):
    lis = ['resized','grey_scale','blured','contrast','rotate','brightnes','negative_filim','edge_detection','sharpnes','adding_frame','mirror_image']
    show_option = st.sidebar.radio("SELECT  FOR VIEW",lis)
    image = Image.open(image)
    if show_option == 'grey_scale':
        col1, col2 = st.columns(2)
        with col1:
            st.title('Original_grey')
            plt.axis('off')
            plt.imshow(image.convert("L"))
            st.pyplot(plt.gcf())
        with col2:
            st.title('resized_grey')
            width = st.sidebar.slider("**select width**", min_value=1, max_value=1000)
            height  = st.sidebar.slider("**select height**", min_value=1, max_value=1000)
            resized =image.resize((height,width))
            plt.axis('off')
            plt.imshow(resized.convert("L"))
            st.pyplot(plt.gcf())
    
    elif show_option == 'resized':
        col1, col2 = st.columns(2)
        
        with col1:
            width = st.sidebar.slider("**select width**", min_value=1, max_value=1000)
            height  = st.sidebar.slider("**select height**", min_value=1, max_value=1000)
            
        with col2:
            resized =image.resize((height,width)) 
            plt.axis('off')
            plt.imshow(resized)
            st.pyplot(plt.gcf())    
        
        
    elif show_option == 'blured':
        col1, col2,col3 ,col4,col5= st.columns(5)
        
        #with col1:
        radius = st.sidebar.slider("**radius**", min_value=0, max_value=10)
            
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            #st.write(image.filter(ImageFilter.GaussianBlur(radius=2)))
            
        with col3:
            plt.axis('off')
            st.title('blured_org')
            plt.imshow(image.filter(ImageFilter.GaussianBlur(radius)))
            st.pyplot(plt.gcf())
    
            
        with col5:
            grey_image = image.convert("L")
            st.title('blured_grey')
            plt.axis('off')
            plt.imshow(grey_image.filter(ImageFilter.GaussianBlur(radius)))
            st.pyplot(plt.gcf())
        #st.write(grey_image.filter(ImageFilter.GaussianBlur(radius=2)))
        
            
    elif show_option == 'contrast':
        col1, col2,col3 ,col4,col5= st.columns(5)
        value = st.sidebar.slider("**enhance**", min_value=0, max_value=10)
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            
            
        with col3:
            st.title('org_contr')
            cont = ImageEnhance.Contrast(image)
            plt.axis('off')
            plt.imshow(cont.enhance(value))
            st.pyplot(plt.gcf())
            
        with col5:
            st.title('grey_con')
            grey_image = image.convert("L")
            cont = ImageEnhance.Contrast(grey_image)
            plt.axis('off')
            plt.imshow(cont.enhance(value))
            st.pyplot(plt.gcf())
        
        #st.write(cont.enhance(2))
    elif show_option == 'rotate':
        col1, col2,col3 ,col4,col5= st.columns(5)
        rotate = st.sidebar.slider("**rotation_degreee**", min_value=0, max_value=360)
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            
            
        with col3:
            st.title('org_rotate')
            plt.axis('off')
            plt.imshow(image.rotate(rotate))
            st.pyplot(plt.gcf())
            #st.write(image.rotate({rotate}))
        with col5:
            st.title('grey_rota')
            grey_image = image.convert("L")
            
            plt.axis('off')
            plt.imshow(grey_image.rotate(rotate))
            st.pyplot(plt.gcf())
        
    elif show_option == 'brightnes':
        col1, col2,col3 ,col4,col5= st.columns(5)
        value = st.sidebar.slider("**enhancement**", min_value=1, max_value=10)
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            
        with col3:
            st.title('org_Brightness')
            plt.axis('off')
            org_bri = ImageEnhance.Brightness(image)
            plt.imshow(org_bri.enhance(value))
            st.pyplot(plt.gcf())
            
        with col5:
            st.title('grey_brit')
            grey_image = image.convert("L")
            grey_bright = ImageEnhance.Brightness(grey_image)
            plt.axis('off')
            plt.imshow(grey_bright.enhance(value))
            st.pyplot(plt.gcf())
       
            
    elif show_option == 'negative_filim':
        col1, col2,col3 ,col4,col5= st.columns(5)
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
        with col3:
            st.title('negative_image')
            plt.axis('off')
            plt.imshow(ImageOps.invert(image))
            st.pyplot(plt.gcf())
            
        with col5:
            st.title('grey_neg')
            grey_image = image.convert("L")
            
            plt.axis('off')
            plt.imshow(ImageOps.invert(grey_image))
            st.pyplot(plt.gcf())
            
            
    elif show_option == 'edge_detection':
        col1, col2,col3 ,col4,col5= st.columns(5)
        with col1:
            st.title('Original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
        
        with col3:
            st.title('edged_image')
        
            plt.axis('off')
            plt.imshow(image.filter(ImageFilter.FIND_EDGES))
            st.pyplot(plt.gcf())
        
        with col5:
            st.title('grey_edged')
            grey_image = image.convert("L")
            
            plt.axis('off')
            plt.imshow(grey_image.filter(ImageFilter.FIND_EDGES))
            st.pyplot(plt.gcf())
            
            
    elif show_option == 'adding_frame':
        col1, col2,col3 ,col4,col5= st.columns(5)
        color = st.selectbox('',['red','green','blue'])
        with col1:
            st.title('originial')
            plt.axis('off')
            plt.imshow(ImageOps.expand(image,40,color))
            st.pyplot(plt.gcf())
            
        with col3:
            st.title('grey_image')
            grey_image = image.convert("L")
            
            plt.axis('off')
            plt.imshow(ImageOps.expand(grey_image,40,color))
            st.pyplot(plt.gcf())
            
            
    elif show_option == 'sharpnes':
        col1, col2,col3 ,col4,col5= st.columns(5)
        value = st.sidebar.slider("**enhancement**", min_value=1, max_value=40)
        with col1:
            st.title('original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            
        with col3:
            st.title('org_sharp')
            plt.axis('off')
            org = ImageEnhance.Sharpness(image)
            plt.imshow(org.enhance(value))
            st.pyplot(plt.gcf())
            
        with col5:
            st.title('grey_sharp')
            grey_image = image.convert("L")
            gre=ImageEnhance.Sharpness(grey_image )
            plt.axis('off')
            plt.imshow(gre.enhance(value))
            st.pyplot(plt.gcf())
            
        
        
        
    elif show_option == 'mirror_image':
        col1, col2,col3 ,col4,col5= st.columns(5)
        with col1:
            st.title('original')
            plt.axis('off')
            plt.imshow(image)
            st.pyplot(plt.gcf())
            
        with col3:
            st.title('org_mirror')
            plt.axis('off')
            plt.imshow(ImageOps.mirror(image))
            st.pyplot(plt.gcf())
            
        with col5:
            st.title('grey_mirror')
            grey_image = image.convert("L")
            plt.axis('off')
            plt.imshow(ImageOps.mirror(grey_image))
            st.pyplot(plt.gcf())
            
#********************************************************************************************************************************************
def nou(extreacted_text):
    tag_mapping = {
     'CC': 'Coordinating Conjunction',
     'CD': 'Cardinal Digit',
     'DT': 'Determiner',
     'EX': 'Existential There',
     'FW': 'Foreign Word',
     'IN': 'Preposition or Subordinating Conjunction',
     'JJ': 'Adjective',
     'JJR': 'Adjective, Comparative',
     'JJS': 'Adjective, Superlative',
     'LS': 'List Item Marker',
     'MD': 'Modal',
     'NN': 'Noun, Singular or Mass',
     'NNS': 'Noun, Plural',
     'NNP': 'Proper Noun, Singular',
     'NNPS': 'Proper Noun, Plural',
     'PDT': 'Predeterminer',
     'POS': 'Possessive Ending',
     'PRP': 'Personal Pronoun',
     'PRP$': 'Possessive Pronoun',
     'RB': 'Adverb',
     'RBR': 'Adverb, Comparative',
     'RBS': 'Adverb, Superlative',
     'RP': 'Particle',
     'SYM': 'Symbol',
     'TO': 'to',
     'UH': 'Interjection',
     'VB': 'Verb, Base Form',
     'VBD': 'Verb, Past Tense',
     'VBG': 'Verb, Gerund or Present Participle',
     'VBN': 'Verb, Past Participle',
     'VBP': 'Verb, Non-3rd Person Singular Present',
     'VBZ': 'Verb, 3rd Person Singular Present',
     'WDT': 'Wh-determiner',
     'WP': 'Wh-pronoun',
     'WP$': 'Possessive Wh-pronoun',
     'WRB': 'Wh-adverb'
     }
    tokens =nltk.word_tokenize(extreacted_text)
    result=nltk.pos_tag(tokens)
    noun=[]
    non_noun=[]
    for i,j in result:
         if j in tag_mapping:
             
             
             if j not in ['PRP','NN']:
                 non_noun.append(i)
         
             else:
                 noun.append(i)
                 
    noun = set(noun)
    non_noun = set(non_noun)
    st.write('noun:',' ,'.join(noun))        
    st.write('non_noun:',' ,'.join(non_noun))
    return result
#****************************************************************************************************************************************************        
def text_extract(image):
    image = Image.open(image)
    extreacted_text = pytesseract.image_to_string(image)
    if extreacted_text is not None:
        nou(extreacted_text)
        return extreacted_text
    else:
        st.write( 'no text data theere ')         
        
#***********************************************************************************************************************************************
if options == 'nlp':
    sample=st.text_input(label='enter the text',max_chars= 100)
    
    lis = ['tokenization','spelling_check','stemming','lemmentization','keywords','sentiment','augmentation','POS']
    show_option = st.radio("SELECT  FOR VIEW",lis)

    if show_option == 'tokenization':
        #token(sample)
        st.success(token(sample))
    elif show_option == 'stemming':
        st.success(stem(sample))
    elif show_option == 'lemmentization':
        st.success(lemm(sample))
    elif show_option == 'keywords':
        st.success(keyword(sample))
    elif show_option == 'sentiment':
        st.success(sentiment(sample))
    elif show_option == 'augmentation':
        lis = ['select','synonyms','antonyms','both']
        option =st.selectbox('',lis)
        if option == 'synonyms':
            st.success(aug_synon(sample))
        elif option == 'antonyms':
            st.success(aug_ant(sample))
        elif option == 'both':
            st.success(both(sample))
            
    elif show_option == 'POS':
        st.success(pos(sample))
    
    elif show_option == 'spelling_check':
        st.success(spell(sample))
#**************************************************************************************************************************************************
elif options == 'image':
    
    lis = ['select','upload and show_image','propertise','modyfy propertise','text_extract']
    options =st.selectbox('',lis)
    uploaded= st.file_uploader("upload here", label_visibility="collapsed", type=["png", "jpeg", "jpg"])
    image = save_card(uploaded)
    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Dell\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe" 
    if options == 'upload and show_image':
        save_card(uploaded)
        
    elif options == 'propertise':
        properties_string = propertise(image)
        st.success(properties_string)
    elif options == 'modyfy propertise':
        convert(image)
        
    elif options == 'text_extract':
        st.success(text_extract(image))
            
        
        
 #***********************************************************************************************************************************************           
elif options == 'prediction':
    
    with st.sidebar:
        
        opt = option_menu("Process",["EDA","Evaluation Metrics", "Prediction"])
    class file_retrive:
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)     
    
    file = file_retrive(df=None) # if data is none
    # load csv file
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        file.df = df    
    ## upload file 
    if opt == "EDA":
        col1,col2 = st.columns(2)
        col1, col2 = st.columns(2)
        button1 = col1.button("Show DataFrame and its visualization")
        button2 = col2.button("Show DataFrame and process")
            
        
        if button1:                
        
            if file.df is not None :
                st.dataframe(file.df)
                st.title('Plot Vizualization')
                def plot_bool(column_name):
                    data= file.df[column_name].value_counts()
                    plt.figure(figsize=(16,9))
                    color = sns.color_palette('husl',len(data))
                    data.plot(kind ='bar',color = color)
                    # Disable the warning about PyplotGlobalUse
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    #plt.axis('off')
                    st.pyplot()
                    
                    return column_name
                def plot_int(column_name):
                    data= file.df[column_name].value_counts()
                    plt.figure(figsize=(16,22))
                    color = sns.color_palette('husl',len(data))
                    data.plot(kind ='bar',color = color)
                    # Disable the warning about PyplotGlobalUse
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    #plt.axis('off')
                    st.pyplot()
                    
                    
                    return column_name
                
               
                st.subheader('bool_columns vizual')
                bool_colum = file.df.select_dtypes(include=['bool']).columns
                for i in bool_colum:
                    # Disable the warning about PyplotGlobalUse
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plot_bool(i)
                    
                st.subheader('int_columns vizual')
                int_colum = file.df.select_dtypes(include=['int']).columns
                for i in int_colum:
                    # Disable the warning about PyplotGlobalUse
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plot_bool(i)
                    
                st.title('Normalization Curve')
                if file.df is not None:
                    # label encoding
                    le = LabelEncoder()
                    for col in file.df.columns:
                        if file.df[col].dtype == 'object' or file.df[col].dtype == 'bool':
                            file.df[col] = le.fit_transform(file.df[col])
                        st.write(f"### {col}")
                        sns.kdeplot(data=file.df[col], fill=True, color='g')
                        st.pyplot()
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                
        if button2:        
            if file.df is not None:
                st.write("1. DataFrame")
                st.dataframe(file.df)
                #1.drop duplicate
                if file.df is not None :
                    st.write("2.Drop Duplicates and NaN Values")
                    file.df = file.df.drop_duplicates()
                    file.df = file.df.dropna()
                    st.dataframe(file.df)
                    st.success("Duplicates and NaN values dropped successfully!")
                #2.statistical summery
                if file.df is not None:
                    st.write("3.Statistical summery")
                    st.text(file.df.describe())
                #3.drop zero    
                if file.df is not None:
                    st.write("4.Drop zero columns")
                    col_zero=[]
                    for i in file.df.columns:
                        per_zeo = (file.df[i]==0).mean()*100
                        col_zero.append((i,per_zeo))
                    
                    z_df = pd.DataFrame(col_zero,columns=['colum','zero_percentage'])
                    
                    st.dataframe(z_df)
                    zero_columns=z_df[z_df['zero_percentage']>90]['colum'].sort_values(ascending=False)
                    
                    file.df=file.df.drop(list(zero_columns),axis=1)
                    st.success('successfully zero columns droped')
                    st.dataframe(file.df)
                # label encoding
                if file.df is not None:
                    st.write("5.Label encoding")
                    le = LabelEncoder()
                    for col in file.df.columns:
                        if file.df[col].dtype == 'object' or file.df[col].dtype == 'bool':
                            file.df[col] = le.fit_transform(file.df[col])
                    st.dataframe(file.df)
                    st.success("Label Encoding completed successfully!")
                
                if file.df is not None:
                    st.write("6.'CO-Relationship")
                    cor = file.df.corr()
                    st.dataframe(cor)
                    st.write('7.Heatmap_vizualization')
                    plt.figure(figsize=(10,5))
                    sns.heatmap(cor,cmap='coolwarm',annot=True,fmt = '.2f')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    
                if file.df is not None:
                    st.write("8.outlier_detection and treated")
                    plt.figure(figsize=(15,8))
                    sns.boxplot(data=file.df)
                    st.pyplot()
                    st.success('outlier detected')
                    st.write('outlier treated using invert transformation')
                    file.df['transactionRevenue'] = file.df['transactionRevenue'].replace(0, file.df[file.df['transactionRevenue'] != 0]['transactionRevenue'].mean())
                    file.df['transactionRevenue'] =1/ file.df['transactionRevenue']
                    plt.figure(figsize=(15,8))
                    sns.boxplot(data=file.df)
                    st.pyplot()
                    st.success('outlier treated successfully!')
                    
                
                    
                if file.df is not None:
                    st.write("9.feature_importance")
                    x =abs(file.df.drop('has_converted',axis = 1))
                    y = file.df['has_converted']
                    rfc = RandomForestClassifier(n_estimators=150)
                    rfc.fit(x,y)
                    
                    rfc_data=pd.DataFrame({
                    "columns":x.columns,
                    "rfc-value": rfc.feature_importances_
                    }).sort_values('rfc-value',ascending=True).head(10)
                    st.write('feature selection using Random forest')
                    st.dataframe(rfc_data)
                    fig = px.pie(rfc_data,
                                 names="columns",
                                 values="rfc-value",
                                 color="rfc-value",
                                 labels={"columns":"columns"},
                                 
                                 title='Pie Chart for feature importance')
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
    elif opt == "Evaluation Metrics":
        st.tilte('Evaluation Metrics')
        st.dataframe(file.df)
        st.subheader('EDA Process:')
        #duplicate,Nan values ,zero values Removed successfully
        file.df=file.df.drop_duplicates()
        zero_columns=['youtube','totals_newVisits','days_since_last_visit','bounces','bounce_rate']
        file.df = file.df.drop(zero_columns,axis = 1)
        file.df =file.df.drop(['latest_medium','target_date'],axis = 1)
        st.write('- duplicate,Nan values ,zero values Removed successfully')
        #values are encoded 
        le = LabelEncoder()
        obj_column = file.df.select_dtypes(include = 'object').columns
        for i in obj_column:
            file.df[i]=le.fit_transform(file.df[i])
        st.write('- values are encoded successfully')
        #co-relation 
        file.df=file.df.drop(['device_deviceCategory','latest_visit_id','visitId_threshold','last_visitId'],axis = 1)
        st.write(' - co-relation find and treated')
        
        #outliesr
        file.df['transactionRevenue'] = file.df['transactionRevenue'].replace(0, file.df[file.df['transactionRevenue'] != 0]['transactionRevenue'].mean())
        file.df['transactionRevenue'] =1/ file.df['transactionRevenue']
        st.write('- outliesr detected and treated')
        
        #random forest feature importance
        x =abs(file.df.drop('has_converted',axis = 1))
        y = file.df['has_converted']
        rfc = RandomForestClassifier(n_estimators=150)
        
        rfc.fit(x,y)
        rfc_data = pd.DataFrame({
        "columns":x.columns,
        "rfc-value": rfc.feature_importances_
        }).sort_values('rfc-value',ascending=True).head(10)
        file.df=file.df.drop(list(rfc_data['columns']),axis = 1)
        st.write('- important features identified using randomforest')
        
        #scaling
        x = file.df.drop('has_converted',axis = 1)
        y = file.df['has_converted']
        train_data,test_data,train_lab,test_lab = train_test_split(x,y,test_size=0.2,random_state= 42)
        sc =StandardScaler()
        sc.fit(train_data)
        train_data =sc.transform(train_data)
        test_data =sc.transform(test_data)
        st.write('scaling data:',train_data)
        st.write('- scaling process done using standardscalar')
        
        #model selection
        st.subheader('Evaluation metrics')
        model = {
        'logistic regression':LogisticRegression(),
        'gaussian':GaussianNB(),
        'supporting vector':SVC(),
        'k_neignbour':KNeighborsClassifier(),
        'decision tree':DecisionTreeClassifier(),
        'extra tree':ExtraTreeClassifier(),
        'randon_forest':RandomForestClassifier(),
        'gradient boost':GradientBoostingClassifier(),
        'ada boost':AdaBoostClassifier(),
        'bagging':BaggingClassifier()
        
        }
        result = []
        for name,md in model.items():
            md.fit(train_data,train_lab)
            train_predi = md.predict(train_data)
            test_predi = md.predict(test_data)
            
            train_acc= accuracy_score(train_lab, train_predi)
            train_f1 = f1_score(train_lab, train_predi,average= 'micro') 
            train_preci = precision_score(train_lab, train_predi,average= 'micro')
            train_recall = recall_score(train_lab, train_predi,average= 'micro')
             
            #***********************************************************
            test_acc= accuracy_score(test_lab, test_predi)
            test_f1 = f1_score(test_lab, test_predi,average= 'micro') 
            test_preci = precision_score(test_lab, test_predi,average= 'micro')
            test_recall = recall_score(test_lab, test_predi,average= 'micro')
            
            
        
            # Append results to the list
            result.append({
                'Model': name,
                'Train_Accuracy': train_acc,
                'Train_F1_Score': train_f1,
                'Train_Precision': train_preci,
                'Train_Recall': train_recall,
                
                'Test_Accuracy': test_acc,
                'Test_F1_Score': test_f1,
                'Test_Precision': test_preci,
                'Test_Recall': test_recall
                
            })
            
        # Create a DataFrame from the list of results
        df_results = pd.DataFrame(result)
        st.dataframe(df_results)
        st.bar_chart(df_results)
        
        model_name =df_results['Model'] 
        acc = df_results['Test_Accuracy']
        
        plt.figure(figsize=(18, 6))
        plt.bar(model_name, acc, color =['blue', 'green', 'red', 'purple','orange','grey','cyan','magenta','yellow','black'])
        plt.ylabel('Accuracy')
        plt.title('Comparison of Classification Model Accuracies')
        plt.ylim(0, 1)
        for i, v in enumerate(acc):
            plt.text(i, v + 0.01, " {:.3f}".format(v), ha='center', color='black')
        st.pyplot()
    if opt == "Prediction":
        with open("rfc.pkl","rb") as mf:
            new_model = pickle.load(mf)
        
        with st.form("user_inputs"):
            with st.container():
                count_session = st.number_input("count_session")
                count_hit = st.number_input("count_hit")
                device_operatingSystem = st.number_input("device_operatingSystem")
                historic_session = st.number_input("historic_session")
                historic_session_page = st.number_input("historic_session_page")
                avg_session_time = st.number_input("avg_session_time")
                avg_session_time_page  = st.number_input("avg_session_time_page ")
                single_page_rate = st.number_input("single_page_rate")
                sessionQualityDim = st.number_input("sessionQualityDim")
                earliest_visit_id = st.number_input("earliest_visit_id")
                earliest_visit_number = st.number_input("earliest_visit_number")
                latest_visit_number = st.number_input("latest_visit_number")
                time_earliest_visit = st.number_input("time_earliest_visit")
                time_latest_visit = st.number_input("time_latest_visit")
                avg_visit_time = st.number_input("avg_visit_time")
                visits_per_day= st.number_input("visits_per_day")
                earliest_source = st.number_input("earliest_source")
                earliest_medium = st.number_input("earliest_medium")
                earliest_keyword = st.number_input("earliest_keyword")
                latest_keyword = st.number_input("latest_keyword")
                num_interactions = st.number_input("num_interactions")
                time_on_site = st.number_input("time_on_site")
                transactionRevenue = st.number_input("transactionRevenue")
                products_array = st.number_input("products_array")
                
            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button:
                test_data = np.array([
                    [
                        count_session, count_hit, device_operatingSystem,historic_session,historic_session_page, avg_session_time,avg_session_time_page , single_page_rate,sessionQualityDim, earliest_visit_id, earliest_visit_number,latest_visit_number, time_earliest_visit,time_latest_visit,avg_visit_time, visits_per_day,earliest_source,earliest_medium, earliest_keyword, latest_keyword,num_interactions, time_on_site, transactionRevenue,products_array
                    ]
                ])
                
                # Convert the data to float
                test_data = test_data.astype(float)
            
                # Make predictions
                predicted = new_model.predict(test_data)[0]
                prediction_proba = new_model.predict_proba(test_data)
            
                # Display the results
                st.write("Prediction:", predicted)
                st.write("Prediction Probability:", prediction_proba)     
     
     
        
        
            
#***********************************************************************************************************************************************
elif options == 'customer recommendation':
    list_options = ['Select one', 'Process Overview','Data', 'Show recommentation']
    option = st.selectbox('', list_options)
    
    data =pd.read_csv("Groceries_dataset.csv")



    data = data.drop_duplicates()
    data['Date']= pd.to_datetime(data['Date'],format='mixed')
    unique_data = data.groupby(['Member_number','Date'])['products'].unique().agg(list).reset_index()
    df = unique_data.sort_values('Member_number',ascending=True)
    df['Rating'] = data['Ratings']
    readed = Reader(line_format='user item rating',rating_scale=(0, 1))
    df['ProductsID'] = df['products'].apply(lambda x: hash(tuple(x)))
    df.fillna(0,inplace=True)


    data_for_sur = Dataset.load_from_df(df[['Member_number','ProductsID','Rating']],readed)
    train_data, test_data = train_test_split(data_for_sur,test_size=0.2)
    model = SVD()
    model.fit(train_data)

        

    def show_recom(cust_id, number_of_rec):
        
        products_overall = df['ProductsID'].unique()   ### Collecting all preduct ID
        procust_bought = df[df['Member_number']==cust_id]['ProductsID'].values ### Finding the customer already bought products ID
        prod_used_for_predict = [i for i in products_overall if i not in procust_bought]
        predict_recom = [(prod, model.predict(cust_id,prod).est) for prod in prod_used_for_predict]
        
        output_recom = [df[df['ProductsID']== i_d]['products'].values[0]   for i_d, a in predict_recom[:number_of_rec]]
        out_list = list(chain.from_iterable(output_recom))
        out_list = list(set(out_list))
        
        return out_list


    if option == 'Process Overview':
        st.write('1.create dataset')
        st.write('2.drop duplicales and fill nun with zeros')
        st.write('3.changing date time')
        st.write('4.Create Surprise dataset')
        st.write('5. Build  model')
        st.write('6. Generate recommendations')


    if option =='Data':
        
        st.write('### Original Data')
        st.write(data)
        

    if option == 'Show recommentation':
        selected_customer = st.selectbox('Select a CustomerId for recommendations:', data['Member_number'].unique())
        
        if st.button('Generate Recommendations'):
            st.subheader(f'Top Recommendations for CustomerId {selected_customer}:')
            result = show_recom(selected_customer, 2)
            for i in result:
                st.write(i)
        
                
    
