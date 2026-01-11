import streamlit as st
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from langchain_community.embeddings import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator
from sklearn.metrics import accuracy_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size:18px !important; }
    .doc-card { padding: 15px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #007bff; }
    .symptom-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 1. GENERATE DOCTOR DATABASE ---
@st.cache_resource
def generate_doctors():
    specialties = ['Cardiologist', 'Dermatologist', 'Gastroenterologist', 'Neurologist', 'General Physician', 'Endocrinologist']
    locations = ['City Heart Center', 'Skin Care Clinic', 'City Hospital', 'Metro Wellness', 'Community Clinic', 'Apollo Wing']
    
    # YOUR CUSTOM DOCTORS (First 4)
    custom_names = [
        'Dr. Harshal Meherkhamb', 
        'Dr. Ojas Jain', 
        'Dr. Wellborn Bar', 
        'Dr. Vedant Wankhede'
    ]
    
    # GENERIC DOCTORS
    generic_names = [
        'Dr. A. Sharma', 'Dr. B. Verma', 'Dr. C. Gupta', 'Dr. D. Iyer', 'Dr. E. Khan', 
        'Dr. F. Singh', 'Dr. G. Das', 'Dr. H. Kaur', 'Dr. I. Mehta', 'Dr. J. Rao',
        'Dr. K. Nair', 'Dr. L. Patel', 'Dr. M. Joshi', 'Dr. N. Reddy', 'Dr. O. Malhotra',
        'Dr. P. Agarwal', 'Dr. Q. Rizvi', 'Dr. R. Dubey', 'Dr. S. Saxena', 'Dr. T. Bose',
        'Dr. U. Chopra', 'Dr. V. Hegde', 'Dr. W. Dsouza', 'Dr. X. Fernandez', 'Dr. Y. Pandey',
        'Dr. Z. Qureshi', 'Dr. A. Mehra'
    ]
    
    all_names = custom_names + generic_names
    
    docs = []
    for i, name in enumerate(all_names):
        spec = specialties[i % len(specialties)]
        loc = locations[i % len(locations)]
        
        docs.append({
            'Doctor_ID': i,
            'Name': name,
            'Specialist': spec,
            'Location': f"{loc}, Block {random.choice(['A','B','C'])}",
            'Time': f"{random.randint(9, 11)}:00 AM - {random.randint(2, 6)}:00 PM",
            'Fee': f"₹{random.choice([500, 800, 1000, 1200, 1500])}"
        })
    return pd.DataFrame(docs)

# --- 2. LOAD MEDICAL DATA & LOGIC ---
@st.cache_resource
def load_system():
    # Load Excel
    df = pd.read_excel('symptoms_with_remedies.xlsx')
    df.fillna('', inplace=True)
    
    # Clean Symptoms
    df['Combined_Symptoms'] = df['Symptom_1'] + " " + df['Symptom_2'] + " " + df['Symptom_3'] + " " + df['Symptom_4']
    df['Clean_Symptoms_List'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].apply(
        lambda x: ', '.join([s.replace('_', ' ').strip() for s in x if s]), axis=1
    )

    # EXTRACT UNIQUE SYMPTOMS FOR DROPDOWN
    all_symptoms = set()
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        unique_list = df[col].dropna().unique().tolist()
        for item in unique_list:
            if item: # if not empty
                all_symptoms.add(item.replace('_', ' ').strip())
    
    sorted_symptoms = sorted(list(all_symptoms))
    
    # Build AI Brain
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    symptom_vectors = embedding_model.embed_documents(df['Combined_Symptoms'].tolist())
    
    # Find Top 5 matches
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(symptom_vectors)
    
    # CALCULATE ACCURACY (Validation Test)
    # We test if the model can find the correct disease for its own rows
    distances, indices = knn.kneighbors(symptom_vectors)
    
    # Check if the Top 1 prediction matches the actual disease
    correct_predictions = 0
    total_samples = len(df)
    
    for i in range(total_samples):
        # indices[i][0] is the closest match to row i (which should be itself)
        predicted_index = indices[i][0]
        if df.iloc[predicted_index]['Disease'] == df.iloc[i]['Disease']:
            correct_predictions += 1
            
    accuracy = (correct_predictions / total_samples) * 100
    
    # Load Doctor DB
    doctors_df = generate_doctors()
    
    return df, knn, embedding_model, doctors_df, sorted_symptoms, accuracy

# Helper: Map Disease to Specialist
def get_specialist_type(disease_name):
    disease_name = disease_name.lower()
    if 'heart' in disease_name: return 'Cardiologist'
    if 'skin' in disease_name or 'fungal' in disease_name or 'rash' in disease_name or 'acne' in disease_name or 'psoriasis' in disease_name or 'pox' in disease_name: return 'Dermatologist'
    if 'stomach' in disease_name or 'abd' in disease_name or 'gerd' in disease_name or 'jaundice' in disease_name or 'vomit' in disease_name: return 'Gastroenterologist'
    if 'paralysis' in disease_name or 'brain' in disease_name or 'migraine' in disease_name: return 'Neurologist'
    if 'diabetes' in disease_name or 'hypothyroid' in disease_name: return 'Endocrinologist'
    return 'General Physician' # Default

# Load System
try:
    df, knn, embedding_model, doctors_df, sorted_symptoms, model_accuracy = load_system()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1 
if 'search_results' not in st.session_state: st.session_state.search_results = None

# --- SIDEBAR (METRICS) ---
st.sidebar.title("📊 Model Performance")
st.sidebar.metric(label="Validation Accuracy", value=f"{model_accuracy:.2f}%")
st.sidebar.write("This metric calculates how accurately the AI maps symptoms back to the correct disease in the dataset.")
st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 Developed by **Harshal & Team**")

# --- 4. APP INTERFACE ---
st.title("🏥 Healthcare Chatbot")
st.markdown("---")

# STEP 1: INPUT (Modified for Multiple Symptoms)
if st.session_state.step == 1:
    st.subheader("👤 Step 1: Select Your Symptoms")
    st.write("You can select multiple symptoms from the list below:")
    
    # MULTI-SELECT DROPDOWN
    selected_symptoms = st.multiselect("Search and select symptoms:", sorted_symptoms)
    
    # OPTIONAL: Text Input for Backup
    st.write("OR")
    user_text_input = st.text_input("Type in Hindi (if you can't find it in the list):")
    
    if st.button("Analyze Condition"):
        query_text = ""
        
        # Priority: Use Dropdown selection if available
        if selected_symptoms:
            query_text = ", ".join(selected_symptoms)
        elif user_text_input:
            # If typing, use translator
            translator = GoogleTranslator(source='auto', target='en')
            query_text = translator.translate(user_text_input)
            
        if query_text:
            with st.spinner("Analyzing combined symptoms..."):
                # Get Top 5 Matches
                query_vector = embedding_model.embed_query(query_text)
                distances, indices = knn.kneighbors([query_vector])
                
                # Store results
                st.session_state.search_results = df.iloc[indices[0]]
                st.session_state.step = 2
                st.rerun()
        else:
            st.warning("Please select at least one symptom or type a description.")

# STEP 2: SYMPTOM VERIFICATION
elif st.session_state.step == 2:
    st.subheader("🔍 Step 2: Verify your Condition")
    st.write("Based on your combination of symptoms, here are the most likely causes:")
    
    matches = st.session_state.search_results
    
    # Prepare options for Radio Button
    options = []
    for i, row in matches.iterrows():
        opt_str = f"**Disease:** {row['Disease']}  \n   *Matches Symptoms:* {row['Clean_Symptoms_List']}"
        options.append(opt_str)
    
    choice = st.radio("Select the closest match:", options)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Condition"):
            selected_index = options.index(choice)
            st.session_state.final_selection = matches.iloc[selected_index]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔙 Go Back"):
            st.session_state.step = 1
            st.rerun()

# STEP 3: REMEDY & SPECIALIST ID
elif st.session_state.step == 3:
    selection = st.session_state.final_selection
    specialist_needed = get_specialist_type(selection['Disease'])
    
    st.subheader(f"🩺 Diagnosis: {selection['Disease']}")
    st.success(f"**💊 Recommended Remedy:** {selection['Remedies']}")
    st.info(f"**ℹ️ Medical Advice:** For this condition, you should consult a **{specialist_needed}**.")
    
    st.write("---")
    st.write("### Would you like to book an appointment now?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 Show Available Doctors"):
            st.session_state.specialist_needed = specialist_needed
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("❌ No, I'm good"):
            st.session_state.step = 1
            st.rerun()

# STEP 4: DOCTOR SELECTION
elif st.session_state.step == 4:
    specialist = st.session_state.specialist_needed
    st.subheader(f"👨‍⚕️ Select a {specialist}")
    
    # FILTER Doctors
    available_doctors = doctors_df[doctors_df['Specialist'] == specialist]
    
    if available_doctors.empty:
        st.warning(f"No specific {specialist} found. Showing General Physicians.")
        available_doctors = doctors_df[doctors_df['Specialist'] == 'General Physician']

    # Dropdown Selection
    doctor_labels = available_doctors.apply(lambda x: f"{x['Name']} | {x['Location']} | Fee: {x['Fee']} | {x['Time']}", axis=1).tolist()
    
    selected_doc_str = st.selectbox("Choose your doctor:", doctor_labels)
    
    if st.button("✅ Confirm Booking"):
        st.session_state.booked_doc = selected_doc_str
        st.session_state.step = 5
        st.rerun()

# STEP 5: FINAL TICKET
elif st.session_state.step == 5:
    st.balloons()
    st.success("✅ Appointment Confirmed Successfully!")
    
    st.markdown(f"""
    <div class="doc-card">
        <h3>🎟️ Booking Receipt</h3>
        <p><strong>Patient Issue:</strong> {st.session_state.final_selection['Disease']}</p>
        <p><strong>Doctor Details:</strong> {st.session_state.booked_doc}</p>
        <p><em>Please arrive 15 minutes before the mentioned time.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Start New Patient"):
        st.session_state.step = 1
        st.rerun()