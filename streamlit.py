import streamlit as st
import requests

# Set the title of the web app
st.title("Text Input and Response App")

# Create a text input field for user query
user_input = st.text_area("Enter your query here:", height=150)

# Create a button to send the query to the API
if st.button("Submit"):
    # Check if the input is not empty
    if user_input:
        # Make a POST request to your FastAPI endpoint
        try:
            response = requests.post("http://localhost:80/generate_response", json={"query": user_input})
            # If the request was successful
            if response.status_code == 200:
                # Extract the response JSON
                result = response.json().get("response", "No response found.")
                st.markdown(result, unsafe_allow_html=True)  # Render the response as HTML
            else:
                st.error("Error: " + str(response.status_code))
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before submitting.")

# Create a button to clear the input and output
if st.button("Clear"):
    user_input = ""
    st.experimental_rerun()  # Refresh the app to clear the response
