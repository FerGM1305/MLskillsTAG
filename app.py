import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

if 'rec' not in st.session_state:
    st.session_state['rec'] = 0

gpt35turbo_csv = "./Machine%20Learning%20engineer_competences_llm.csv"
gpt35turbo_df = pd.read_csv(gpt35turbo_csv)

golden_csv = "./golden_ksa.csv"
golden_df = pd.read_csv(golden_csv)

nesta_csv = "./nesta_ksa.csv"
nesta_df = pd.read_csv(nesta_csv)

# Load next record from dataframe
def load_next_record(gpt35turbo_df, index):
    gpt35turbo_record = gpt35turbo_df["competences_llm"][index]
    return gpt35turbo_record



if st.sidebar.button('Next'):
    st.session_state.rec += 1
    if st.session_state.rec >= len(gpt35turbo_df):
        st.session_state.rec = 0

if st.session_state.rec > 0:
    if st.sidebar.button('Prev'):
        st.session_state.rec -= 1
        if st.session_state.rec < 0:
            st.session_state.rec = 0

cleaned_items = []
for item in gpt35turbo_df["competences_llm"][st.session_state.rec].strip().split("\n"):
    if ". " in item:
        cleaned_items.append(item.split(". ")[1])

job_desc = gpt35turbo_df["job_description"][st.session_state.rec].strip()



def updateRec():
    st.session_state.rec = st.session_state.theSliderProgress

st.sidebar.write(f"Record {st.session_state.rec+1} of {len(gpt35turbo_df)}")

sliderProgress = st.sidebar.slider('p', 1, len(gpt35turbo_df), st.session_state.rec+1, on_change=updateRec, key="theSliderProgress")

showAnnotated = st.sidebar.checkbox('Show annotated')

data_as_csv= golden_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(label="Download",data=data_as_csv,file_name="ksa.csv",mime="text/csv")




with st.container():

    # Load next record
    gpt35turbo_record = load_next_record(gpt35turbo_df, st.session_state.rec)

    st.header("Machine Learning Engineer")
    golden_df2 = pd.DataFrame()
    
    o_col1, o_col2 = st.columns([1,1])

    with o_col1:

        st.subheader("Competences list")
        i = 0
        for item in cleaned_items:
            if item in golden_df["Text"].values:
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
            elif item in nesta_df["Text"].values:
                if showAnnotated:
                    index = nesta_df.index[nesta_df["Text"] == item].tolist()[0]
                    if nesta_df.at[index, 'Label'] == "Knowledge":
                        st.write(item + " 🟢 (nesta)")
                    elif nesta_df.at[index, 'Label'] == "Skill":
                        st.write(item + " 🔵 (nesta)")
                    elif nesta_df.at[index, 'Label'] == "Ability": 
                        st.write(item + " 🟡 (nesta)")
                    else:
                        st.write(item + " ⚪️ (nesta)")
                
            else:
                st.write(item + " 🔴")
                col1, col2, col3, col4, col5 = st.columns([1,1,1,1,8])
                with col1:
                    if st.button("K", key = item + "k" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Knowledge"], 'Text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
                with col2:
                    if st.button("S", key = item + "s" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Skill"], 'Text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
                with col3:
                    if st.button("A", key = item + "a" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Ability"], 'Text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()
                with col4:
                    if st.button("O", key = item + "o" + str(i)):
                        new_row = pd.DataFrame({'Label': ["Other"], 'Text': [item], 'Desc': [item]})
                        golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                        golden_df2.to_csv("golden_ksa.csv", index=False)
                        st.rerun()

            i += 1

    with o_col2:
        st.subheader("Original job post description")
        st.write(job_desc)
        st.divider()
        st.page_link(gpt35turbo_df["job_google_link"][st.session_state.rec], label="Job google link", icon="📄")
        st.page_link(gpt35turbo_df["job_apply_link"][st.session_state.rec], label="Job apply link", icon="📋")
    
