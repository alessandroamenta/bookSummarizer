import streamlit as st
import textwrap
from bookChat import generate_summary
import time

# Set the page title and icon
st.set_page_config(page_title="Book Summarizer", page_icon="📚", layout="wide")

def main():
    st.title("📚🤖🤓 AI Book Summarizer")
    st.write("Ever thought, 'Why read when GPT can do it for me?' Well, you're in the right place! Upload your book and get the gist without the fuss.")

    # Sidebar for settings, How This Works section, and dark mode switch
    with st.sidebar:
        st.subheader("Settings 🛠️")
        st.write("Enter your OpenAI API key:")
        openai_api_key = st.text_input("API Key", type="password")
        st.write("💡 This process is super cheap! However, please make sure you have access to GPT-4, as we use it for the final summary.")

        # How This Works section in the sidebar
        with st.expander("🔍 How This Works (and Why It's Cheap)"):
            st.write("""
            Here's a simple breakdown of how we turn your book into a neat summary without breaking the bank:
            
            1. **Splitting & Embeddings**: The entire book is split into smaller sections and then turned into embeddings, which is a cost-effective process.
            2. **Clustering**: These embeddings are grouped based on their similarity. For each group, we pick the most representative embedding and map it back to its corresponding text section.
            3. **Summarization**: We use a cheaper model, GPT-3.5, to get a summary of these main parts.
            4. **Combining Summaries**: In the end, we use GPT-4 to join these summaries into one smooth summary.
            
            So, while we do use the fancier GPT-4 model, it's only at the end, which helps keep costs down. Just make sure you have access to GPT-4 when you put in your OpenAI key for the best results.
            """)

    # Main content area for file uploader
    st.write('📖 Upload any book (PDF or EPUB) to generate a summary.')
    uploaded_file = st.file_uploader("Choose a book file", type=['pdf', 'epub'])

    if uploaded_file:
        # Create a placeholder for the button
        button_placeholder = st.empty()

        # Display the button in the placeholder
        generate_button = button_placeholder.button("Sum it up!")

        if generate_button:
            # Disable the button by clearing the placeholder
            button_placeholder.empty()
            # Error handling
            if not openai_api_key:
                st.error("Oops! Looks like you forgot to enter your OpenAI API key in the sidebar. 🤔")
                return
            if uploaded_file.type not in ["application/pdf", "application/epub+zip"]:
                st.error("Hmm... That doesn't seem to be a valid PDF or EPUB file. Please try again with a supported format. 📄")
                return

            with st.spinner("Summarizing your book... Sip your coffee or take a quick stretch. We're on it, chief! 🚀"):
                summary = generate_summary(uploaded_file, openai_api_key)

            wrapped_summary = textwrap.fill(summary, width=80)
            st.subheader('Summary:')
            st.code(wrapped_summary, language="txt")
            st.success("Summary generated successfully! 🎉")

if __name__ == "__main__":
    main()