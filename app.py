import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

if 'rec' not in st.session_state:
    st.session_state['rec'] = 0

gpt35turbo_csv = "./Machine%20Learning%20engineer_competences_llm.csv"
gpt35turbo_df = pd.read_csv(gpt35turbo_csv)

golden_csv = "./golden_ksa.csv"
golden_df = pd.read_csv(golden_csv)

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
    else:
        pass

with st.container():

    # Load next record
    gpt35turbo_record = load_next_record(gpt35turbo_df, st.session_state.rec)

    st.header("Machine Learning Engineer")
    golden_df2 = pd.DataFrame()
    
    st.subheader("Competences list")
    for item in cleaned_items:
        if item in golden_df["Text"].values:
            pass
            # index = golden_df.index[golden_df["Text"] == item].tolist()[0]
            # if golden_df.at[index, 'Label'] == "Knowledge":
            #     st.write(item + " 🟢")
            # elif golden_df.at[index, 'Label'] == "Skill":
            #     st.write(item + " 🔵")
            # elif golden_df.at[index, 'Label'] == "Ability":
            #     st.write(item + " 🟡")
            # else:
            #     st.write(item + " ⚪️")
            
        else:
            st.write(item + " 🔴")
            col1, col2, col3, col4, col5 = st.columns([1,1,1,1,8])
            with col1:
                if st.button("K", key = item + "k"):
                    new_row = pd.DataFrame({'Label': ["Knowledge"], 'Text': [item], 'Desc': [item]})
                    golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                    golden_df2.to_csv("golden_ksa.csv", index=False)
                    st.rerun()
            with col2:
                if st.button("S", key = item + "s"):
                    new_row = pd.DataFrame({'Label': ["Skill"], 'Text': [item], 'Desc': [item]})
                    golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                    golden_df2.to_csv("golden_ksa.csv", index=False)
                    st.rerun()
            with col3:
                if st.button("A", key = item + "a"):
                    new_row = pd.DataFrame({'Label': ["Ability"], 'Text': [item], 'Desc': [item]})
                    golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                    golden_df2.to_csv("golden_ksa.csv", index=False)
                    st.rerun()
            with col4:
                if st.button("O", key = item + "o"):
                    new_row = pd.DataFrame({'Label': ["Other"], 'Text': [item], 'Desc': [item]})
                    golden_df2 = pd.concat([golden_df, new_row], ignore_index=True)
                    golden_df2.to_csv("golden_ksa.csv", index=False)
                    st.rerun()
    

st.sidebar.write(f"Record {st.session_state.rec+1} of {len(gpt35turbo_df)}")
data_as_csv= golden_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(label="Download",data=data_as_csv,file_name="ksa.csv",mime="text/csv")

