import streamlit as st
import pandas as pd
from itertools import zip_longest
import math
from thefuzz import fuzz
from thefuzz import process
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(layout="wide")

if 'rec' not in st.session_state:
    st.session_state['rec'] = 0

if 'occ' not in st.session_state:
    st.session_state['occ'] = 0


golden_csv = "./golden_ksa.csv"
golden_df = pd.read_csv(golden_csv)

occ_csv = "./infocomm_occupations.csv"
occ_df = pd.read_csv(occ_csv)


def load_next_occ(_occ_df, index):
    _occ = (_occ_df["Occupation"][index],_occ_df["OccupationQS"][index])
    return _occ

occ = load_next_occ(occ_df,st.session_state['occ'])


gpt35turbo_csv = f"./infocomm_jsearch_jobposts/3-abril-2024-processed/{occ[1]}_competences_llm.csv"
#gpt35turbo_csv = f"../infocomm/3-abril-2024-processed/{occ[1]}_competences_llm.csv"
#gpt35turbo_csv = "../infocomm_jsearch_jobposts/3-abril-2024-processed/Data%20science%20engineer_competences_llm.csv"
gpt35turbo_df = pd.read_csv(gpt35turbo_csv)

# Load next record from dataframe
def load_next_record(gpt35turbo_df, index):
    gpt35turbo_record = gpt35turbo_df["competences_llm"][index]
    return gpt35turbo_record



cleaned_items = []
for item in gpt35turbo_df["competences_llm"][st.session_state.rec].strip().split("\n"):
    if ". " in item:
        cleaned_items.append(item.split(". ")[1])

job_desc = gpt35turbo_df["job_description"][st.session_state.rec].strip()



def updateRec():
    st.session_state.rec = st.session_state.theSliderProgress

def updateOcc():
    st.session_state.occ = st.session_state.theSliderProgressOcc
    if st.session_state.occ >= len(occ_df):
        st.session_state.occ = 0

sliderProgressOcc = st.sidebar.slider(f"Occ {st.session_state.occ+1} of {len(occ_df)}", 1, len(occ_df), st.session_state.occ+1, on_change=updateOcc, key="theSliderProgressOcc")
if st.sidebar.button('Next occ', key='nextOcc'):
    st.session_state.occ += 1
    if st.session_state.occ >= len(occ_df):
        st.session_state.occ = 0
    st.rerun()

if st.session_state.occ > 0:
    if st.sidebar.button('Prev occ'):
        st.session_state.occ -= 1
        if st.session_state.occ < 0:
            st.session_state.occ = 0
        st.rerun()


st.sidebar.title(occ[0])

sliderProgress = st.sidebar.slider(f"Record {st.session_state.rec+1} of {len(gpt35turbo_df)}", 1, len(gpt35turbo_df), st.session_state.rec+1, on_change=updateRec, key="theSliderProgress")
if st.sidebar.button('Next rec'):
    st.session_state.rec += 1
    if st.session_state.rec >= len(gpt35turbo_df):
        st.session_state.rec = 0
    st.rerun()

if st.session_state.rec > 0:
    if st.sidebar.button('Prev rec'):
        st.session_state.rec -= 1
        if st.session_state.rec < 0:
            st.session_state.rec = 0
        st.rerun()

showAnnotated = st.sidebar.checkbox('Show annotated')

st.sidebar.write("Annotated KSA: " + str(len(golden_df)))

data_as_csv= golden_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(label="Download",data=data_as_csv,file_name="ksa.csv",mime="text/csv")



# ngrams 

def ngrams(input_str, n):
    # Generate n-grams from the input string
    input_str = input_str.lower()
    return zip_longest(*[input_str[i:] for i in range(n)], fillvalue='')

 
def character_ngram_similarity(word1, word2, n):
    if not(isinstance(word1, str)) or not(isinstance(word2, str)):
        return 0
    # Calculate character n-gram similarity
    ngrams_word1 = set(ngrams(word1, n))
    ngrams_word2 = set(ngrams(word2, n))
 
    common_ngrams = ngrams_word1 & ngrams_word2
    similarity = len(common_ngrams) / max(len(ngrams_word1), len(ngrams_word2))
 
    return similarity

def getSimilarNgrams(_item):
    _sim = set()
    
    max_sim = 0
    for golden in golden_df["Text"].values:
        sim = character_ngram_similarity(golden, _item, 2)
        if sim > 0.5:
            max_sim += 1
            if max_sim > 2:
                break
            index = golden_df.index[golden_df["Text"] == golden].tolist()[0]
            #_sim.add(golden)
            _ksa = golden_df.at[index, 'Label']
            #Get standard text for button
            goldenStandardText = golden_df.at[index, 'Standard text']
            if _ksa != "Other":
                _sim.add((_ksa,goldenStandardText))

    max_sim = 0
    for goldenStandardText in golden_df["Standard text"].values:
        sim = character_ngram_similarity(golden, _item, 2)
        if sim > 0.5:
            max_sim += 1
            if max_sim > 2:
                break
            index = golden_df.index[golden_df["Standard text"] == goldenStandardText].tolist()[0]
            _ksa = golden_df.at[index, 'Label']
            if _ksa != "Other":
                _sim.add((_ksa,goldenStandardText))

    #If the text is inside any "Standar text" string then add it once:
    max_sim = 0
    for goldenStandardText in golden_df["Standard text"].values:
        if goldenStandardText in _item:
            max_sim += 1
            if max_sim > 1:
                break
            index = golden_df.index[golden_df["Standard text"] == goldenStandardText].tolist()[0]
            _ksa = golden_df.at[index, 'Label']
            if _ksa != "Other":
                _sim.add((_ksa,goldenStandardText))


    return list(_sim)


# thefuzz

def fuzz_similarity(word1, word2):
    #return fuzz.ratio(word1, word2)
    #return fuzz.partial_token_sort_ratio(word1, word2)
    return fuzz.token_sort_ratio(str(word1).lower(), str(word2).lower())

def getSimilarFuzz(_item):
    _sim = set()

    #get max 3 similar items from 'Text' and 'Standard text'

    max_sim = 0
    for golden in golden_df["Text"].values:
        sim = fuzz_similarity(golden, _item)
        if sim > 50:
            max_sim += 1
            if max_sim > 3:
                break
            #_sim.add(golden + " (" + str(sim) + ")")
            index = golden_df.index[golden_df["Text"] == golden].tolist()[0]
            _ksa = golden_df.at[index, 'Label']
            #Get standard text for button
            goldenStandardText = golden_df.at[index, 'Standard text']
            if _ksa != "Other":
                _sim.add((_ksa,goldenStandardText))
    return list(_sim)


with st.container():

    # Load next record
    gpt35turbo_record = load_next_record(gpt35turbo_df, st.session_state.rec)

    golden_df2 = pd.DataFrame()
    

    _ii = 0

    pendingItems = []

    for item in cleaned_items:
        if item in golden_df["Text"].values:
            _ii += 1
        else:
            pendingItems.append(item)

    _sofar = len(cleaned_items) - len(pendingItems)
    st.subheader(f"Competences list   {_sofar}/{len(cleaned_items)}")

    i = 0


    # Get first n cleaned_items items:
    # TODO: try pagination, if not, increase batch_size from time to time untill all reached
    batch_size = 1

    if len(pendingItems) >= batch_size:
        pendingItems = pendingItems[0:batch_size]
    else:
        pendingItems = pendingItems[0:len(pendingItems)]

    for item in pendingItems:
        # Your code to process each item goes here
        similar_ngrams = []
        similar_thefuzz = []

        if item in golden_df["Text"].values: # Javascript != JavaScript
            if showAnnotated:
                index = golden_df.index[golden_df["Text"] == item].tolist()[0]
                if golden_df.at[index, 'Label'] == "Knowledge":
                    st.write(item + " 🟢 (golden)")
                elif golden_df.at[index, 'Label'] == "Skill":
                    st.write(item + " 🔵 (golden)")
                elif golden_df.at[index, 'Label'] == "Ability": 
                    st.write(item + " 🟡 (golden)")
                else:
                    st.write(item + " ⚪️ (golden)")
                st.divider()
        else:
            st.write(item + " 🔴")
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,3,4])
            with col1:
                with stylable_container(
                    "greenK",
                    css_styles="""
                    button {
                        background-color: #0ead69;
                        color: white;
                    }""",
                ):
                    if st.button("K", key = item + "k" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Knowledge"], 'Text': [item], 'Standard text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
            with col2:
                with stylable_container(
                    "blueS",
                    css_styles="""
                    button {
                        background-color: #00a6fb;
                        color: white;
                    }""",
                ):
                    if st.button("S", key = item + "s" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Skill"], 'Text': [item], 'Standard text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
            with col3:
                with stylable_container(
                    "yellowA",
                    css_styles="""
                    button {
                        background-color: #ffd60a;
                        color: white;
                    }""",
                ):
                    if st.button("A", key = item + "a" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Ability"], 'Text': [item], 'Standard text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
            with col4:
                with stylable_container(
                    "grayO",
                    css_styles="""
                    button {
                        background-color: #666;
                        color: white;
                    }""",
                ):
                    if st.button("O", key = item + "o" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Other"], 'Text': [item], 'Standard text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
            
            with col5:
                ksa_option = st.selectbox(
                    'a',
                    ('🍭', 'K', 'S', 'A', 'O'),
                    label_visibility="collapsed",
                    key = "ksa_input" + str(i)
                )
                
            with col6:
                new_ksa = st.text_input(
                    "Enter new KSA",
                    label_visibility="collapsed",
                    placeholder="Add new",
                    key = item + "ksa_input" + str(i)
                )
                
            with col7:
                if st.button("Add", key = item + "btn" + str(i)):                        
                    # Temp hack for streamlit selectbox until value != label is available.
                    _ksa_type = ""
                    if ksa_option == "K":
                        _ksa_type = "Knowledge"
                    elif ksa_option == "S":
                        _ksa_type = "Skill"
                    elif ksa_option == "A":
                        _ksa_type = "Ability"
                    elif ksa_option == "O":
                        _ksa_type = "Other"
                    
                    #st.write(f"{_ksa_type}, {new_ksa}")

                    new_row = pd.DataFrame({'Label': [_ksa_type], 'Text': [item], 'Standard text': [new_ksa], 'Desc': [new_ksa]})
                    golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                    golden_df2.to_csv("golden_ksa.csv", index=False)
                    st.rerun()
                
                
            #ngrams
            #st.write("ngrams (2)")
            similar_ngrams = getSimilarNgrams(item)
            for similar_item in  similar_ngrams:
                i+=1
                with stylable_container(
                    "ngram",
                    css_styles="""
                    button {
                        background-color: #ff9b54;
                        color: black;
                    }""",
                ):
                    if st.button(similar_item[1], key = str(similar_item[1]) + str(i)):
                        #mark it
                        new_row = pd.DataFrame({'Label': [similar_item[0]], 'Text': [item], 'Standard text': [similar_item[1]], 'Desc': [similar_item[1]]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()

            #thefuzz
            #st.write("thefuzz")
            similar_thefuzz = getSimilarFuzz(item)
            for similar_item in  similar_thefuzz:
                i+=1
                with stylable_container(
                    "thefuzz",
                    css_styles="""
                    button {
                        background-color: #ff7f51;
                        color: black;
                    }""",
                ):
                    if st.button(similar_item[1], key = str(similar_item[1]) + str(i)):
                        #mark it
                        new_row = pd.DataFrame({'Label': [similar_item[0]], 'Text': [item], 'Standard text': [similar_item[1]], 'Desc': [similar_item[1]]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()

            st.divider()

        i += 1

   

with st.container():
    st.subheader("Original job post description")
    st.write(job_desc)
    st.divider()
    st.page_link(gpt35turbo_df["job_google_link"][st.session_state.rec], label="Job google link", icon="📄")
    st.page_link(gpt35turbo_df["job_apply_link"][st.session_state.rec], label="Job apply link", icon="📋")
