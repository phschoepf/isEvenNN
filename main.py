import streamlit as st
from is_even_nn import *


st.set_page_config(layout="centered", page_icon="🤖", page_title="Is it even?")
st.title("IsEvenNN")

st.write("A neural network that predicts if a number is even!")

left, right = st.columns(2)

form = left.form("input")
number_input = form.text_input("Your number:")
submit = form.form_submit_button("Submit number")

if submit:
    evenNN = IsEvenNN()
    evenNN.net.load_state_dict(torch.load("isEvenModel.pt"))
    try:
        res = evenNN.predict_single(number_input)
        st.balloons()
        if res:
            right.success(f"{number_input} is even!")
        else:
            right.error(f"{number_input} is not even!")
    except Exception as e:
        right.error(f"Did not understand input {number_input} ({e.__class__.__name__})")
