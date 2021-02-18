#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


# In[4]:


dataset = pd.read_csv("roo_data.csv")


# In[5]:


dataset.iloc[:,-1]


# In[6]:


data = dataset.iloc[:,:-1].values
label = dataset.iloc[:,-1].values
l=12
len(data[0])


# In[7]:


dataset.iloc[:,14:38]


# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[9]:


labelencoder = LabelEncoder()


# In[10]:


for i in range(14,38):
    data[:,i] = labelencoder.fit_transform(data[:,i])
data[:5]


# In[11]:


dataset.iloc[:,:14]


# In[12]:


data[:5,14:]


# In[13]:


from sklearn.preprocessing import Normalizer


# In[14]:


data1=data[:,:14]


# In[15]:


normalized_data = Normalizer().fit_transform(data1)
print(normalized_data.shape)


# In[16]:


normalized_data


# In[17]:


data2=data[:,14:]
data2.shape


# In[18]:


df1 = np.append(normalized_data,data2,axis=1)
df1.shape


# In[19]:


## adding headers

X1 = pd.DataFrame(df1,columns=['Acedamic percentage in Operating Systems', 'percentage in Algorithms',
       'Percentage in Programming Concepts',
       'Percentage in Software Engineering', 'Percentage in Computer Networks',
       'Percentage in Electronics Subjects',
       'Percentage in Computer Architecture', 'Percentage in Mathematics',
       'Percentage in Communication skills', 'Hours working per day',
       'Logical quotient rating', 'hackathons', 'coding skills rating',
       'public speaking points', 'can work long time before system?',
       'self-learning capability?', 'Extra-courses did', 'certifications',
       'workshops', 'talenttests taken?', 'olympiads',
       'reading and writing skills', 'memory capability score',
       'Interested subjects', 'interested career area ', 'Job/Higher Studies?',
       'Type of company want to settle in?',
       'Taken inputs from seniors or elders', 'interested in games',
       'Interested Type of Books', 'Salary Range Expected',
       'In a Realtionship?', 'Gentle or Tuff behaviour?',
       'Management or Technical', 'Salary/work', 'hard/smart worker',
       'worked in teams ever?', 'Introvert'])

X1.head()


# In[20]:


label1=dataset.iloc[:,-1]


# In[21]:


label1


# In[22]:


arr=np.unique(label1)


# In[23]:


Dict = {}
for i in range(0,34): 
    Dict[arr[i]] = i


# In[24]:


Dict


# In[25]:


label = labelencoder.fit_transform(label)
print(len(label))


# In[26]:


y=pd.DataFrame(label,columns=["Suggested Job Role"])
y.head()


# # Decision Tree Classifier

# In[27]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=10) 


# In[28]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[29]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[30]:


y_pred = clf.predict(X_test)


# In[31]:


y_pred


# In[32]:


cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)


# In[33]:


print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*1000)


# # With entropy

# In[34]:


clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 10)
clf_entropy.fit(X_train, y_train)


# In[35]:


entropy_y_pred=clf_entropy.predict(X_test)


# In[36]:


cm_entopy = confusion_matrix(y_test,entropy_y_pred)


# In[37]:


entropy_accuracy = accuracy_score(y_test,entropy_y_pred)


# In[38]:


print("confusion matrics=",cm_entopy)
print("  ")
print("accuracy=",l*entropy_accuracy*100)


# # Support Vector Machines

# In[39]:


from sklearn import svm


# In[40]:


clf = svm.SVC()
clf.fit(X_train, y_train)


# In[41]:


svm_y_pred = clf.predict(X_test)


# In[42]:


svm_cm = confusion_matrix(y_test,svm_y_pred)
svm_accuracy = accuracy_score(y_test,svm_y_pred)


# In[43]:


print("confusion matrics=",svm_cm)
print("  ")
print("accuracy=",l*svm_accuracy*100)


# # XGBoost

# In[94]:


X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=10) 


# In[95]:


X_train.shape


# In[96]:


X_train=pd.to_numeric(X_train.values.flatten())


# In[97]:


X_train=X_train.reshape((14000,38))


# In[98]:


from xgboost import XGBClassifier


# In[99]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[100]:


xgb_y_pred = clf.predict(X_test)


# In[101]:


xgb_cm = confusion_matrix(y_test,xgb_y_pred)
xgb_accuracy = accuracy_score(y_test,xgb_y_pred)


# In[102]:


print("confusion matrics=",xgb_cm)
print("  ")
print("accuracy=",l*xgb_accuracy*100)


# # CHATBOT

# In[44]:


import io
import random
import string
import warnings
import numpy as np
import pandas as pd


# In[45]:


from sklearn import decomposition
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


'''
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
'''


# In[48]:


import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('popular', quiet=True)
f=open('chatbot.txt','r',errors = 'ignore')
RawData=f.read()
RawData = RawData.lower()


# In[49]:


sent_tokens = nltk.sent_tokenize(RawData)# converts to list of sentences 
word_tokens = nltk.word_tokenize(RawData)# converts to list of words


# In[50]:


lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[51]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Oy' mate! can i help you out today?", "Hello buddy! can i help you out today?", "*nods*", "Saheli at your service", "hello friend. can i help you out today?", "i am glad that you are talking to me"]

def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[52]:


#Various Responses out of which a random one will be picked
QUESTIONAIRE_Qs = ("What is your percentage in Operating Systems?" , "What about in Data Structures and Algorithms?" , "What about in Programming Concepts?" , "What about in in Software Engineering?", "What is your percentage in Computer Networks?", "What is your percentage in Electronics subjects?" , "What is your percentage in Computer Architecture?" , "What is your percentage in Mathematics?" , "What is your percentage in Communication skills?" , "How many hours working per day?" , "Rate your Logical quotient out of 10!" , " Number of hackathons attended?" ," Rate your Coding skills." , " Rate your Public speaking skills out of 10" , " Can you work for long time before system?" , "Can you learn things on your own?" , "Do you have interest in extra-curriculars?" , "What interests you more, Machine Learning or app development?" , "Choose one: Cloud computing, database security, game development, hacking, testing, data science!" , " Did you take any talent tests?" , " Did you attend any olympiads?" , "What best describes your Reading and writing skills: Excellent, medium or poor" , "What best describes your memory skills: Excellent, medium or poor" , "Name the most interesting subject you are interested in CS" , "Do you see yourself as a developer or a Business process analyst?" , "Would you prefer job or higherstudies?","What type of company do you want to work in?", "Have you taken any inputs from any seniors?", "Are you interested in games?","What kind of books are you into? because that matters too! :P", "Would you prefer work or salary?", "Do you have other commitments like clubs and departments?", "What do you think best describes you best: gentle or stubborn?", "Which one of the following are you more inclined towards: Management or technical?", "Would you prefer work or salary?", "Are you a hard worker or a smart worker?", "Do you like working in teams?", "Are you an introvert?")
GREAT_WORK = ("That's amazing! ", "I'm proud of you! ", "wow! ", "Damn! Nice dude!", "Hey that's pretty awesome if you ask me! ","Noice! ","Lovely! ","Such a good kid! ",)
MEDIUM_WORK = ("hey not bad! ","Okay alright! ", "Shouldn't be that big a problem! ", "Oh acha, okay! ", "That's where most of us lie, right? haha! ")
POOR_WORK = ("Don't worry! ", "Ouch!! ", "OOPS! ", "Somebody wasn't paying attention, huh? ", "Oh no..", "That's bad dude! ", "It's okay, you can work on that! ",)
FINAL_ANSWER = ("That's the end of this quiz! After a lot of thinking, I think you should become a ", "We've come to the end of the quiz! My research suggests that you should become a ", "In my opinion, you should actually consider becoming a ", " My Final say is, You're best suited for the role of a ", "You must be wondering what I found out! You should definitely be a ", "That's the end of this quiz! After extenive research of your personal choices and academic backgroun, you should definitely be a ")


# In[ ]:


flag = True
 
relative_response = " "

in_quiz = False
collect_answer = False
percentage_question = True
question_no = -1

ANSWERS = []

print("SAHELI: Hey")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("SAHELI: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("SAHELI: "+ "My name is Saheli. I aim to help you out with career guidance. If you want to exit, type Bye!" + greeting(user_response) + "! I believe you're confused about your ideal job role. If you want to take a quick questionnaire, type 'start'")

        if(user_response=='start'):
            in_quiz = True
            print("SAHELI: Great, Let's begin the quiz then")
            
        if(user_response.isdigit()):
            if(percentage_question):
                if(int(user_response) < 40):
                    relative_response = random.choice(POOR_WORK)
                if((int(user_response) < 60) & (int(user_response) > 40)):
                    relative_response = random.choice(MEDIUM_WORK)
                if(int(user_response) > 60):
                    relative_response = random.choice(GREAT_WORK)
            else:
                if(int(user_response) > 6):
                    relative_response = random.choice(GREAT_WORK)
                if((int(user_response) < 6) & (int(user_response) > 4)):
                    relative_response = random.choice(MEDIUM_WORK)
                if(int(user_response) < 4):
                    relative_response = random.choice(POOR_WORK)
        if(in_quiz):
            ####store the given answer in an array"
            ######################################
            ANSWERS.append(user_response)
            
            question_no = question_no + 1
            if(question_no == 38):
                in_quiz = False
                ###Call model with stored value"
                ###############################
                ###############################
                print("Saheli: WELL DONE!!")
                ANSWERS.remove('start')
                for i in range(14,38):
                    ANSWERS[14:38] = labelencoder.fit_transform(ANSWERS[14:38])
                for i in range(0,14):
                    ANSWERS[i]= int(ANSWERS[i])
                ANSWERS_ARRAY = np.asarray(ANSWERS)
                data123=data
                ANS=ANSWERS_ARRAY
                ANS.reshape(1,38)
                ANS1=np.reshape(ANS,(1,38))
                new_data = np.append(data123,ANS1,axis=0)
                new_data_for_normalization = new_data[:,:14]
                normalized_data_for_prediction = Normalizer().fit_transform(new_data_for_normalization)
                print(normalized_data_for_prediction.shape)
                remaining_data = new_data[:,14:]
                new_df = np.append(normalized_data_for_prediction,remaining_data,axis=1)
                prediction_array = new_df[20000,:]
                prediction_array.reshape(1,-1)
                prediction_array2d= [prediction_array]
                clf = svm.SVC()
                clf.fit(X_train, y_train)
                predicted_value = clf.predict(prediction_array2d)
                
                
                print(random.choice(FINAL_ANSWER) + list(Dict.keys())[list(Dict.values()).index(predicted_value[0])] )
                data123=data
                ANS=ANSWERS_ARRAY
                #predict the output of the model using ANSWERS
                
                print("Saheli: Thank you for taking this quiz, please say 'bye' to exit")
                
                
            if(question_no > 8):
                percentage_question = False
            print("SAHELI: " + relative_response + QUESTIONAIRE_Qs[question_no])

            collect_answer = True
            relative_response = ""
            
    else:
        flag=False
        print("SAHELI: Bye! take care..")
        
  
        


# In[54]:


print(ANSWERS)


# In[55]:


print(len(ANSWERS))


# In[56]:


for i in range(14,38):
    ANSWERS[14:38] = labelencoder.fit_transform(ANSWERS[14:38])


# In[57]:


print(ANSWERS)


# In[58]:


for i in range(0,14):
    ANSWERS[i]= int(ANSWERS[i])


# In[59]:


print(ANSWERS)


# In[60]:


type(ANSWERS)


# In[61]:


type(data)


# In[62]:


data.shape


# In[63]:


ANSWERS_ARRAY = np.asarray(ANSWERS)


# In[66]:


ANSWERS_ARRAY


# In[67]:


type(ANSWERS_ARRAY)


# In[68]:


data123=data
ANS=ANSWERS_ARRAY


# In[69]:


data123.shape


# In[70]:


ANS.shape


# In[71]:


ANS.reshape(1,38)


# In[72]:


ANS.shape


# In[73]:


ANS1=np.reshape(ANS,(1,38))


# In[74]:


ANS1.shape


# In[75]:


new_data = np.append(data123,ANS1,axis=0)


# In[76]:


new_data


# In[77]:


new_data_for_normalization = new_data[:,:14]


# In[78]:


new_data_for_normalization.shape


# In[79]:


normalized_data_for_prediction = Normalizer().fit_transform(new_data_for_normalization)
print(normalized_data_for_prediction.shape)


# In[80]:


normalized_data_for_prediction


# In[81]:


remaining_data = new_data[:,14:]
remaining_data.shape


# In[82]:


new_df = np.append(normalized_data_for_prediction,remaining_data,axis=1)
new_df.shape


# In[83]:


prediction_array = new_df[20000,:]


# In[84]:


prediction_array


# In[85]:


prediction_array.shape


# In[86]:


prediction_array.reshape(1,-1)


# In[87]:


prediction_array2d= [prediction_array]


# In[88]:


clf = svm.SVC()
clf.fit(X_train, y_train)
predicted_value = clf.predict(prediction_array2d)


# In[89]:


predicted_value


# In[90]:


predicted_value[0]


# In[91]:


y.iloc[46,:]


# In[92]:


key_list = list(Dict.keys()) 
val_list = list(Dict.values())


# In[93]:


print(list(Dict.keys())[list(Dict.values()).index(predicted_value[0])]) 


# 
