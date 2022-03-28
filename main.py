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
    evenNN.net.load_state_dict(torch.load("isEvenModel.pt", map_location=torch.device("cpu")))
    try:
        res, conf = evenNN.predict_single(number_input)
        message = f'{number_input} is {"" if res else "not "}even! (confidence {conf:.3f})'
        if res:
            right.success(message)
        else:
            right.error(message)
    except IndexError as ie:
        right.error(f"Failed to process \"{number_input}\" ({ie.__class__.__name__}: {ie})")
    except ValueError as ve:
        right.error(f"Failed to process \"{number_input}\" ({ve.__class__.__name__}: {ve})")
    except Exception as e:
        right.error(f"Failed to process \"{number_input}\" (Unknown Error)")
