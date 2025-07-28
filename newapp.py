import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------ #
# ✅ App Config and Setup
# ------------------------------ #
st.set_page_config(page_title="SmartLearnPath", layout="wide")
st.title("📘 SmartLearnPath: AI-Powered Learning Path Generator")
st.markdown("🔍 *A BERT-based Personalized Learning Assistant for Your Career Goals*")

# ------------------------------ #
# ✅ Load Data & Model
# ------------------------------ #
@st.cache_data
def load_courses():
    df = pd.read_csv("coursera_course_dataset_v3.csv")
    df = df.dropna(subset=["course_description"])
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------ #
# ✅ Recommendation Logic
# ------------------------------ #
def recommend_courses(user_profile, courses, model, top_n=6):
    user_embedding = model.encode([user_profile])
    course_embeddings = model.encode(courses["course_description"].tolist())
    similarities = cosine_similarity(user_embedding, course_embeddings)[0]
    courses["Similarity"] = similarities
    return courses.sort_values(by="Similarity", ascending=False).head(top_n)

# ------------------------------ #
# ✅ App UI: Input Section
# ------------------------------ #
st.sidebar.header("🧑‍🎓 Enter Your Learning Preferences")

goal = st.sidebar.text_area("🎯 Career Goal", placeholder="e.g. I want to become a Data Analyst")
skills = st.sidebar.text_input("🛠 Current Skills", placeholder="e.g. Python, SQL, Excel")
target_role = st.sidebar.selectbox("💼 Target Job Role", ["Data Analyst", "Backend Developer", "Project Manager", "AI Engineer", "UX Designer"])
hours = st.sidebar.slider("🕒 Weekly Learning Hours", 1, 30, 6)

submit = st.sidebar.button("🚀 Generate My Learning Path")

# ------------------------------ #
# ✅ On Submit: Recommend Path
# ------------------------------ #
if submit and goal and skills:
    with st.spinner("Generating personalized recommendations..."):
        df, model = load_courses(), load_model()
        user_profile = f"{goal}. My current skills are {skills}. I want to become a {target_role}."

        results = recommend_courses(user_profile, df, model)

        st.subheader("📚 Recommended Courses")
        for i, row in results.iterrows():
            with st.expander(f"{row['Title']}"):
                st.markdown(f"- 🌐 **Organization:** {row['Organization']}")
                st.markdown(f"- 🧠 **Skills Covered:** {row['Skills']}")
                st.markdown(f"- 🎓 **Difficulty:** `{row['Difficulty']}`")
                st.markdown(f"- ⭐ **Ratings:** `{row['Ratings']}`")
                st.markdown(f"- 👨‍🎓 **Students Enrolled:** {row['course_students_enrolled']}")
                st.markdown(f"- ⏳ **Duration:** {row['Duration']}")
                st.markdown(f"- 🔗 [Go to Course]({row['course_url']})")
                st.markdown(f"- 📈 **Similarity Score:** `{row['Similarity']:.2f}`")

        # Missing Skills Placeholder
        st.divider()
        st.subheader("🧩 Skill Gap Insight")
        st.markdown(f"Based on your goal to become a **{target_role}**, we recommend strengthening:")
        st.info("📝 Placeholder: Use skill gap logic here to compute actual missing skills")

        # Export Button
        st.subheader("📥 Export Your Personalized Learning Path")
        export_df = results[["Title", "Skills", "Difficulty", "Ratings", "course_url", "Duration"]]
        st.download_button(
            label="📁 Download as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="SmartLearnPath_Recommendations.csv",
            mime="text/csv"
        )

        # Chat Assistant Placeholder
        st.divider()
        st.subheader("🤖 Ask SmartLearn AI (coming soon)")
        query = st.chat_input("Ask anything about your skills, path, or courses...")
        if query:
            st.info("This is a placeholder. You can integrate GPT via OpenAI API here.")

elif submit:
    st.warning("Please fill in all fields in the sidebar to get your learning path.")
